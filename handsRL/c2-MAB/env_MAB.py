# 实现K=10的多臂老虎机，拉动每根拉杆的奖励分布服从Bernouli Distribution 有p的概率获得奖励为1，有1-p的概率获得奖励为0

# Implement a multi-armed slot machine with K=10, and the reward Distribution of pulling each tie rod follows the Bernouli Distribution. The probability of getting a reward is 1 with p, and the probability of getting a reward is 0 with 1-p

import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    def __init__(self, K):
        self.probs = np.random.uniform(size=K) # 随机生成K个0-1的数作为拉动每根杠杆的概率
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.K = K
    def step(self,K):
        # 根据概率返回，其实就是一个if else条件判断句
        if np.random.rand() < self.probs[K]:
            return 1 
        else:
            return 0
np.random.seed(1)
K = 10
bandit_10_arm = BernoulliBandit(K)
print('Get a random multi-arms bandit with %d arms' % K)
print('The best prob index is %d ,best prob is %.4f' % (bandit_10_arm.best_idx,bandit_10_arm.best_prob))

# -- General Solver Framework---
class Solver:
    def __init__(self,bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K) # 每次拉杆的尝试次数
        self.regret = 0 # 当前步的累积懊悔
        self.actions = [] # 维护一个列表，记录每一步的动作
        self.regrets = [] # 维护一个列表，记录每一步的懊悔
    
    def update_regret(self,k):
        # calculate cumulated regret and save it,k is the rod number selected for this action
        # 计算累积懊悔并保存，k为本次动作选择的拉杆编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)
    
    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆，由每个具体的策略实现
        raise NotImplementedError
    
    def run(self, num_steps):
        # 运行一定次数，num_steps 运行的总次数
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)
def plot_results(solvers, solvers_names):
    # 生成累积懊悔随时间变化的图像。输入的solvers是一个列表,列表中每个元素是一个特定策略
    # solver_names也是一个列表，存储每个策略的名称
    for idx,solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list,solver.regrets, label = solvers_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()



