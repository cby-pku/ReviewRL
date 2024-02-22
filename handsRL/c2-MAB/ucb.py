from env_MAB import *
import numpy as np
import matplotlib.pyplot as plt

class UCB(Solver):
    def __init__(self, bandit, coef, init_prob = 1.0):
        super(UCB,self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef
    
    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / (2 * (self.counts + 1)))
        k = np.argmax(ucb)
        r =self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


np.random.seed(1)
coef = 1 # 控制不确定系数
UCB_solver = UCB(bandit_10_arm, coef)
UCB_solver.run(5000)
print('上置信界算法的累积懊悔为: ', UCB_solver.regret)
plot_results([UCB_solver],['UCB'])
plt.savefig('UCB.png')
plt.close()
