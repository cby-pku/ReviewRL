from env_MAB import *
import numpy as np
import matplotlib.pyplot as plt

class ThompsonSampling(Solver):
    def __init__(self, bandit):
        super(ThompsonSampling,self).__init__(bandit)
        self._a = np.ones(self.bandit.K) # 列表，表示每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.K) # 列表，表示每根拉杆奖励为0的次数

    def run_one_step(self):
        samples = np.random.beta(self._a,self._b)
        k = np.argmax(samples)
        r = self.bandit.step(k)

        self._a[k] += r
        self._b[k] += (1-r)
        return k

np.random.seed(1)
thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(5000)
print('Thompson Sampling cumulative regret: ', thompson_sampling_solver.regret)
plot_results([thompson_sampling_solver],['ThompsonSamling'])
plt.savefig('thompson.png')
plt.close()