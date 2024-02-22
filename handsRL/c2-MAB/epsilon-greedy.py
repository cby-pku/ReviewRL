from env_MAB import *
import numpy as np
import matplotlib.pyplot as plt

# -- epsilon-greedy ---
class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon = 0.01, init_prob = 1.0):
        super(EpsilonGreedy,self).__init__(bandit)
        self.epsilon = epsilon
        # 初始化所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob]*self.bandit.K)
    
    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k) # 得到本次动作的奖励
        # 增量式更新
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k
    

np.random.seed(1)
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
epsilon_greedy_solver.run(5000)
print('epsilon-greedy cumulative regrets is : ',epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver],['EpsilonGreedy'])
plt.savefig("Epsilon-Greedy.png")
plt.close()

# different epsilon and observe its curve
np.random.seed(0)
epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
epsilon_greedy_solver_list = [EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons]
epsilon_greedy_solver_names = ['epsilon={}'.format(e) for e in epsilons]
for solver in epsilon_greedy_solver_list:
    solver.run(5000)
plot_results(epsilon_greedy_solver_list,epsilon_greedy_solver_names)
plt.savefig('Dif-EpsilonGreedy.png')
plt.close()
# 结论：在MAB+epsilon-greedy的例子中，累积懊悔会线性增长，这是因为self.regret的定义导致的，一旦出现了探索，就会增加固定值的regret

# --- 尝试随时间衰减的epsilon-greedy algorithm
class DecayingEpsilonGreedy(Solver):
    ''' epsilon值随着时间改变: 反比例衰减'''
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy,self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0
    
    def run_one_step(self):
        self.total_count += 1
        if np.random.random()<1 / self.total_count :
            k = np.random.randint(0, self.bandit.K)
        else :
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r-self.estimates[k])

        return k

np.random.seed(1)
decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000)
print('epsilon值衰减的贪婪算法的累积懊悔为: ',decaying_epsilon_greedy_solver.regret)
plot_results([decaying_epsilon_greedy_solver],['DecayingEpsilonGreedy'])
plt.savefig('DecayingEpsilonGreedy.png')
plt.close()

# 结论：随着时间做反比例的epsilon-greedy decaying算法可以让累积懊悔与时间的关系变成次线性的
