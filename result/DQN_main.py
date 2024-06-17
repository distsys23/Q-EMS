import sys
import os

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

import torch
import datetime
import numpy as np
import pickle

from utils import save_results, make_dir
from utils import plot_rewards
from DQN_model import DQN
from DQN_env import *
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间

class Config:
    '''超参数
    '''
    def __init__(self):
        ################################## 环境超参数 ###################################
        self.algo_name = 'DQN'  # 算法名称
        self.env_name = 'micro_grid'  # 环境名称
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.seed = 10  # 随机种子，置0则不设置随机种子
        self.train_eps = 1000  # 训练的回合数
        self.test_eps = 10  # 测试的回合数
        self.train_day = 0
        ################################################################################
        ################################## 算法超参数 ###################################
        self.gamma = 1  # 强化学习中的折扣因子
        self.epsilon_start = 0.90  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01  # e-greedy策略中的终止epsilon
        self.epsilon_decay = 0.95  # e-greedy策略中epsilon的衰减率
        self.lr = 0.0003  # 学习率
        self.memory_capacity = 4000  # 经验回放的容量
        self.batch_size = 200  # mini-batch SGD中的批量大小
        self.target_update = 10  # 目标网络的更新频率
        self.hidden_dim = 256  # 网络隐藏层
        ################################################################################
        ################################# 保存结果相关参数 ################################
        self.result_path = './result/'  # 保存结果的路径
        # self.model_path = curr_path + "/outputs/" + self.env_name + \
        #                   '/' + curr_time + '/models/'  # 保存模型的路径
        #"20221221-154606good"
        self.save = True  # 是否保存图片
        ################################################################################
def env_agent_config(cfg):
    ''' 创建环境和智能体
    '''
    env = MicroGridEnv() # 创建环境
    state_dim = env.observation_space.shape[0]  # 状态维度
    action_dim = env.action_space.n  # 动作维度
    print(state_dim, action_dim, "\n")
    agent = DQN(state_dim, action_dim, cfg)  # 创建智能体
    if cfg.seed != 0:  # 设置随机种子
        torch.manual_seed(cfg.seed)
        #env.seed(cfg.seed)
        np.random.seed(cfg.seed)
    return env, agent
def train(cfg, env, agent):
    '''
    训练
    '''
    print('开始训练!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有time_step的滑动平均奖励
    ep_reward = []  # 记录所有time_step的奖励
    for i_ep in range(cfg.train_eps):
        ep_reward1 = 0
        state = env.reset_all(day=cfg.train_day)  # 重置环境，返回初始状态
        while True:
            action = agent.choose_action(state)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            agent.memory.push(state, action, reward,
                              next_state, done)  # 保存transition
            state = next_state  # 更新下一个状态
            agent.update()  # 更新智能体
            ep_reward1 += reward
            # ep_reward.append(reward) # 累加奖励

            if done:
                break
        ep_reward.append(ep_reward1)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward1)
        else:
            ma_rewards.append(ep_reward1)
        if (i_ep + 1) % cfg.target_update == 0:  # 智能体目标网络更新
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        if (i_ep + 1) % 10 == 0:
            print('回合：{}/{}'.format(i_ep + 1, cfg.train_eps))
    print('完成训练！')
    env.close()
    return ep_reward, ma_rewards


def test(cfg, env, agent):
    print('开始测试!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    ############# 由于测试不需要使用epsilon-greedy策略，所以相应的值设置为0 ###############
    cfg.epsilon_start = 0.0  # e-greedy策略中初始epsilon
    cfg.epsilon_end = 0.0  # e-greedy策略中的终止epsilon
    ################################################################################
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录一回合内的奖励
        state = env.reset_all(day=i_ep)  # 重置环境，返回初始状态
        while True:
            action = agent.choose_action(state)  # 选择动作
            next_state, reward, done, _ = env.step(action,test=True)  # 更新环境，返回transition
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            env.render(test_day=i_ep, display=[0,1,1,1])
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f"回合：{i_ep + 1}/{cfg.test_eps}，奖励：{ep_reward:.1f}")
    print(rewards)
    print('完成测试！')
    env.close()
    return rewards, ma_rewards


if __name__ == "__main__":
    cfg = Config()
    # 训练
    env, agent = env_agent_config(cfg)
    rewards, ma_rewards = train(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='DQN_train',
                 path=cfg.result_path)  # 保存结果
    plot_rewards(rewards, ma_rewards, cfg, tag="train")  # 画出结果
    # 测试
    env, agent = env_agent_config(cfg)
    rewards, ma_rewards = test(cfg, env, agent)
    print("平均奖励", np.mean(rewards))
    save_results(rewards, ma_rewards, tag='DQN_test',
                 path=cfg.result_path)  # 保存结果
    plot_rewards(rewards, ma_rewards, cfg, tag="test")  # 画出结果
