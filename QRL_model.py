import tensorflow as tf
import tensorflow_quantum as tfq
import subprocess
import numpy as np
import cirq, sympy

tf.get_logger().setLevel('ERROR')

def get_gpu_info():
    try:
        return subprocess.check_output(["nvidia-smi"]).decode("utf-8")
    except Exception as e:
        return str(e)
    
def one_qubit_rotation(qubit, symbols):
    """
    Returns Cirq gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [cirq.rx(symbols[0])(qubit),
            cirq.ry(symbols[1])(qubit),
            cirq.rz(symbols[2])(qubit)]

def entangling_layer(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops

def generate_circuit(qubits, n_layers):
    """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
    # Number of qubits
    n_qubits = len(qubits)

    # Sympy symbols for variational angles
    params = sympy.symbols(f'theta(0:{3*(n_layers+1)*n_qubits})')
    params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))

    # Sympy symbols for encoding angles
    inputs = sympy.symbols(f'x(0:{n_layers})'+f'_(0:{n_qubits})')
    inputs = np.asarray(inputs).reshape((n_layers, n_qubits))

    # Define circuit
    circuit = cirq.Circuit()
    for l in range(n_layers):
        # Variational layer
        circuit += cirq.Circuit(one_qubit_rotation(q, params[l, i]) for i, q in enumerate(qubits))
        circuit += entangling_layer(qubits)
        # Encoding layer
        circuit += cirq.Circuit(cirq.rx(inputs[l, i])(q) for i, q in enumerate(qubits))

    # Last varitional layer
    circuit += cirq.Circuit(one_qubit_rotation(q, params[n_layers, i]) for i,q in enumerate(qubits))

    return circuit, list(params.flat), list(inputs.flat)

class ReUploadingPQC(tf.keras.layers.Layer):
    """
    Performs the transformation (s_1, ..., s_d) -> (theta_1, ..., theta_N, lmbd[1][1]s_1, ..., lmbd[1][M]s_1,
        ......., lmbd[d][1]s_d, ..., lmbd[d][M]s_d) for d=input_dim, N=theta_dim and M=n_layers.
    An activation function from tf.keras.activations, specified by `activation` ('linear' by default) is
        then applied to all lmbd[i][j]s_i.
    All angles are finally permuted to follow the alphabetical order of their symbol names, as processed
        by the ControlledPQC.
    """
    def __init__(self, qubits, n_layers, observables, activation="linear", name="re-uploading_PQC"):
        super(ReUploadingPQC, self).__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)
        
        circuit, theta_symbols, input_symbols = generate_circuit(qubits, n_layers)

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )
        lmbd_init = tf.ones(shape=(self.n_qubits * self.n_layers,))
        self.lmbd = tf.Variable(
            initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas"
        )
        # Define explicit symbol order.
        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])
        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)        

    def call(self, inputs):
        # inputs[0] = encoding data for the state.
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)
        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)
        return self.computation_layer([tiled_up_circuits, joined_vars])
    
class Rescaling(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(Rescaling, self).__init__()
        self.input_dim = input_dim
        self.w = tf.Variable(
            initial_value=tf.ones(shape=(1,input_dim)), dtype="float32",
            trainable=True, name="obs-weights")
    def call(self, inputs):
        return tf.math.multiply((inputs+1)/2, tf.repeat(self.w,repeats=tf.shape(inputs)[0],axis=0))    
    
def generate_model_Qlearning(n_qubits, n_layers, n_actions, target):
    """
    Generates a Keras model for a data re-uploading PQC Q-function approximator.
    """
    qubits = cirq.GridQubit.rect(1, n_qubits)
    ops = [cirq.Z(q) for q in qubits]
    observables = [ ops[0]*ops[1]*ops[2],ops[1]*ops[2]*ops[3],ops[2]*ops[3]*ops[4],ops[3]*ops[4]*ops[5],ops[4]*ops[5]*ops[6],
                    ops[0]*ops[1]*ops[3],ops[0]*ops[1]*ops[4],ops[0]*ops[1]*ops[5],ops[0]*ops[1]*ops[6],ops[0]*ops[2]*ops[3],
                    ops[0]*ops[2]*ops[4],ops[0]*ops[2]*ops[5],ops[0]*ops[2]*ops[6],ops[0]*ops[3]*ops[4],ops[0]*ops[3]*ops[5],
                    ops[0]*ops[3]*ops[6],ops[0]*ops[4]*ops[5],ops[0]*ops[4]*ops[6],ops[0]*ops[5]*ops[6],ops[1]*ops[2]*ops[4],
                    ops[1]*ops[2]*ops[5],ops[1]*ops[2]*ops[6],ops[1]*ops[3]*ops[4],ops[1]*ops[3]*ops[5],ops[1]*ops[3]*ops[6],
                    ops[1]*ops[4]*ops[5],ops[1]*ops[4]*ops[6],ops[1]*ops[5]*ops[6],ops[2]*ops[3]*ops[5],ops[2]*ops[3]*ops[6],
                    ops[2]*ops[4]*ops[5],ops[2]*ops[4]*ops[6],ops[2]*ops[5]*ops[6],ops[3]*ops[4]*ops[6],ops[3]*ops[5]*ops[6],
                    ops[0]*ops[1]*ops[2]]
    # observables = [ ops[0]*ops[1],ops[1]*ops[2],ops[2]*ops[3],ops[3]*ops[4],ops[4]*ops[5],ops[5]*ops[6]]   #observables = [ops[i] * ops[i + 1] for i in range(n_qubits - 1)] # Z_0*Z_1 for action 0 and Z_2*Z_3 for action 1
    # while len(observables) < int(n_actions):
    #     # 将 observables 中的每个元素重复一遍并添加到列表中
    #     observables.extend(observables)
    # # 截取前36个元素
    # observables = observables[:int(n_actions)]
    input_tensor = tf.keras.Input(shape=(len(qubits), ), dtype=tf.dtypes.float32, name='input')
    re_uploading_pqc = ReUploadingPQC(qubits, n_layers, observables, activation='tanh')([input_tensor])
    process = tf.keras.Sequential([Rescaling(len(observables))], name=target*"Target"+"Q-values")
    # 这里的Rescaling返回的对象是一个层而非张量
    Q_values = process(re_uploading_pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=Q_values)
    return model
    
@tf.function   
def QDQN_test(model, env, test_eps):
    print('-------------------开始测试!---------------------')
    ################################################################################
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(test_eps):
        ep_reward = 0  # 记录一回合内的奖励
        state = env.reset_all(seed = i_ep)  # 重置环境，返回初始状态
        while True:
            q_vals = model([tf.convert_to_tensor([state])])
            action = int(tf.argmax(q_vals[0]).numpy())  # 选择 Q-values 最大的动作
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
        print(f"回合：{i_ep }/{test_eps - 1}，奖励：{ep_reward:.1f}")
    print(rewards)
    print('完成测试！')
    env.close()
    return rewards, ma_rewards

            