# ############## 动作空间缩小为36 ###############

import sys
import os
import torch
import datetime
from utils import save_results, make_dir
from utils import plot_rewards, plot_losses
from QEMS_env import *
import tensorflow as tf
import numpy as np
import time
from QRL_model import generate_model_Qlearning, QDQN_test, get_gpu_info

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

tf.get_logger().setLevel('ERROR')
# 启用 Eager Execution 模式
tf.config.run_functions_eagerly(True)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_virtual_device_configuration(
#                 gpu,
#                 [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
#     except RuntimeError as e:
#         print(e)

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class Config:
    '''超参数test
    '''

    def __init__(self):
        ################################## 环境超参数 ###################################
        self.algo_name = 'QRL'  # 算法名称
        self.env_name = 'micro_grid'  # 环境名称
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.seed = 10  # 随机种子，置0则不设置随机种子
        self.train_eps = 500  # 训练的回合数
        self.test_eps = 10  # 测试的回合数
        self.train_seed = 0
        self.Baseline = ['DQN', 'Fixed']
        self.Baseline_num = 2
        ################################################################################

        ################################## 算法超参数 ###################################
        self.gamma = 1  # 强化学习中的折扣因子
        self.epsilon_start = 0.90  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01  # e-greedy策略中的终止epsilon
        self.epsilon_decay = 0.95  # e-greedy策略中epsilon的衰减率
        self.memory_capacity = 4000  # 经验回放的容量
        self.batch_size = 200  # mini-batch SGD中的批量大小
        self.target_update = 10  # 目标网络的更新频率
        self.n_actions = 36  # Number of action_dim
        ################################################################################

        ################################# 保存结果相关参数 ################################
        self.result_path = './result/'  # 保存结果的路径
        # self.model_path = curr_path + "/outputs/" + self.env_name + \
        #                   '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片
        ################################################################################


# ############################ QRL相关模型参数 ################################
n_qubits = 7  # Dimension of the state vectors
n_layers = 3  # Number of layers in the PQC
n_actions = 36  # Number of action_dim
opt_in = 0.00025
opt_var = 0.0001
opt_out = 0.0018

model = generate_model_Qlearning(n_qubits, n_layers, n_actions, False)
model_target = generate_model_Qlearning(n_qubits, n_layers, n_actions, True)  # tf.keras.Model
model_target.set_weights(model.get_weights())  # 同样的初始化权重

optimizer_in = tf.keras.optimizers.Adam(learning_rate=opt_in, amsgrad=True)
optimizer_var = tf.keras.optimizers.Adam(learning_rate=opt_var, amsgrad=True)
optimizer_out = tf.keras.optimizers.Adam(learning_rate=opt_out, amsgrad=True)
w_in, w_var, w_out = 1, 0, 2


@tf.function
def Q_learning_update(state, action, reward, next_state, model, cfg):
    '''states: 当前状态的张量。
    actions: 执行的动作的张量。
    rewards: 获得的奖励的张量。
    next_states: 下一个状态的张量。
    model: 当前模型，用于预测 Q 值。
    gamma: 折扣因子。
    n_actions: 动作的数量。'''
    # 将输入数据转换为 TensorFlow 张量
    state = tf.expand_dims(tf.convert_to_tensor(state), axis=0)
    action = tf.expand_dims(tf.convert_to_tensor(action), axis=0)
    reward = tf.expand_dims(tf.convert_to_tensor(reward), axis=0)
    next_state = tf.expand_dims(tf.convert_to_tensor(next_state), axis=0)
    # 获取目标模型的 Q 值
    Q_target = model_target([next_state])
    target_q_values = reward + (cfg.gamma * tf.reduce_max(Q_target, axis=1))

    # 创建动作对应的 one-hot 编码
    masks = tf.one_hot(action, cfg.n_actions, axis=-1)
    # 使用梯度带计算损失
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        q_values = model([state])
        q_values_masked = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        loss = tf.keras.losses.Huber()(target_q_values, q_values_masked)

    # for i, var in enumerate(model.trainable_variables):
    #     print(f"Variable {i}: {var.name} {var}")

    # 计算梯度并使用优化器更新模型参数
    grads = tape.gradient(loss, model.trainable_variables)
    # 对梯度进行裁剪,防止梯度爆炸
    grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]
    for optimizer, w in zip([optimizer_in, optimizer_var, optimizer_out], [w_in, w_var, w_out]):
        optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])

    return loss


def train(cfg, env):
    ''' 训练
    '''
    TrainStartT = time.time()
    print('-------------------开始训练!---------------------')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    loss_list = []
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    eu_reward = 0
    replay_memory = []
    epsilon = cfg.epsilon_start
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录一回合内的奖励
        # 每个回合结束从缓冲池中随机选取一个批量的数据进行模型更新
        state = env.reset_all(seed=cfg.train_seed)  # 随机生成一个HEMS状态

        while True:
            # Sample action
            coin = np.random.random()
            if coin > epsilon:
                q_vals = model([tf.convert_to_tensor([state])])
                action = int(tf.argmax(q_vals[0]).numpy())  # 选择 Q-values 最大的动作
            else:
                action = np.random.choice(cfg.n_actions)
            next_state, reward, done, _ = env.step(action, test=False)  # 更新环境，返回transition
            reward = np.float32(reward)
            replay_memory.append({'state': state,
                                  'action': action,
                                  'next_state': next_state,
                                  'reward': reward})
            if len(replay_memory) >= cfg.memory_capacity:
                del replay_memory[0]

            loss = Q_learning_update(state, action, reward, next_state, model, cfg)  # 每一步更新一次
            loss_list.append(loss)

            state = next_state
            ep_reward += reward
            # print(i_ep)
            if done:  # 每个episode完成
                break

        # Update target model
        if (i_ep + 1) % cfg.target_update == 0:
            model_target.set_weights(model.get_weights())
        i_ep += 1

        if i_ep % 20 == 0:
            loss_mean = round(np.mean(loss_list[-20:]), 4)
            loss_var = round(np.var(loss_list[-20:]), 4)
            print("last[", i_ep, "]iterations loss mean:", loss_mean, "var:", loss_var)

        # Decay epsilon
        epsilon = max(epsilon * cfg.epsilon_decay, cfg.epsilon_end)

        eu_reward += ep_reward
        rewards.append(ep_reward)
        # 计算滑动平均奖励
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
            eu_reward = 0
        # if (i_ep + 1) % 10 == 0:
        #     print('回合：{}/{}, 奖励：{}'.format(i_ep + 1, cfg.train_eps, ep_reward))
        print('回合：{}/{}, 奖励：{}'.format(i_ep + 1, cfg.train_eps, ep_reward))
    print('完成训练！')

    TrainEndT = time.time()
    TrainT = TrainEndT - TrainStartT
    print("训练时间：", TrainT)
    env.close()
    return rewards, ma_rewards, loss_list


def test(cfg, env):
    TestStartT = time.time()
    print('开始测试!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards, ma_rewards = QDQN_test(model, env, cfg.test_eps)
    print('完成测试！')

    TestEndT = time.time()
    TestT = TestEndT - TestStartT
    print("测试时间：", TestT)

    env.close()
    return rewards, ma_rewards


if __name__ == "__main__":
    cfg = Config()

    # 训练
    env = MicroGridEnv()  # 创建环境
    # env, agent = env_agent_config(cfg)
    gpu_info = get_gpu_info()
    rewards, ma_rewards, loss_list = train(cfg, env)
    save_results(rewards, ma_rewards, tag='QEMS_train',
                 path=cfg.result_path)  # 保存结果
    plot_rewards(rewards, ma_rewards, cfg, tag="train")  # 画出奖励函数结果
    # plot_losses(loss_list, cfg, algo="QRL", tag='train')    # 画出损失函数结果

    # 测试
    rewards, ma_rewards = test(cfg, env)
    print("平均奖励", np.mean(rewards))
    save_results(rewards, ma_rewards, tag='test',
                 path=cfg.result_path)  # 保存结果

    plot_rewards(rewards, ma_rewards, cfg, tag="QEMS_test")  # 画出结果