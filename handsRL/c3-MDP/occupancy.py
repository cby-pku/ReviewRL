# 近似估计占用度量，设置一个较大的采样轨迹最大值的长度，然后采样很多次，用状态动作对出现的频率估计实际概率
import numpy as np
def occupancy(episodes, s, a, timestep_max, gamma):
    '''计算动作状态对(s,a)出现的频率，以此来估算策略的占用度量'''
    rho = 0
    total_times = np.zeros(timestep_max) # 记录每个时间步t各被经历过几次
    occur_times = np.zeros(timestep_max) # 记录(s_t,a_t) = (s,a)的次数
    for episode in episodes:
        for i in range(len(episode)):
            (s_opt, a_opt, r ,s_next) = episode[i]
            total_times[i] += 1
            if s == s_opt and a == a_opt:
                occur_times[i] += 1
    for i in reversed(range(timestep_max)):
        if total_times[i]:
            rho += gamma**i * occur_times[i] / total_times[i]
    return (1-gamma) * rho