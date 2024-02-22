import numpy as np
np.random.seed(0)
# 定义状态转移概率矩阵P
P = [
    [0.9, 0.1, 0.0, 0.0, 0.0 ,0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0 ,0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0 ,0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
P =np.array(P)

# 定义奖励函数
rewards = [-1,-2,-2,10,1,0]
gamma = 0.5 # 定义折扣因子
# 给定一条序列，计算从某个索引（起始状态）开始到序列最后（终止状态）得到的回报
def compute_return (start_index,chain,gamma):
    G = 0
    for i in reversed(range(start_index,len(chain))):
        G = gamma * G + rewards[chain[i] - 1]
    return G
# 一个状态序列 
chain = [1, 2, 3, 6]
start_index = 0
G = compute_return(start_index,chain,gamma)
print(f"Return of the chain: {G}")

# 利用解析解求解小型规模的MRP

def compute(P,rewards,gamma,states_num):
    rewards = np.array(rewards).reshape((-1,1)) #将reward转化为列向量的形式

    value = np.dot(np.linalg.inv(np.eye(states_num,states_num) - gamma * P), rewards)
    return value

V = compute(P, rewards, gamma, 6)
print(f'Every state value in MRP \n {V}')

# Figure 3.4 p27 provide a easy markov chain , it originally from David Silver's RL Course so I omitted the code

