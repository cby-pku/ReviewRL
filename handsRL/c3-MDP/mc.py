# example code for Monte Carlo Method
import numpy as np
def join(str1,str2):
    return str1 + '-' + str2
def sample(MDP, Pi, timestep_max, number):
    '''采样函数,策略Pi, 限制最长时间步timestep_max,总共序列数number'''
    S,A,P,R,gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)] #记得排除终止状态，这里是s5
        # 当前状态为终止状态或者时间步太长时，一次采样结束
        while s!='s5' and timestep <= timestep_max:
            timestep += 1
            rand, temp = np.random(), 0
            # 在状态s下根据策略选择动作
            for a_opt in A:
                temp += Pi.get(join(s,a_opt), 0)
                if temp >rand:
                    a = a_opt
                    r = R.get(join(s,a), 0)
                    break
            rand, temp = np.random.rand(),0
            # 根据状态转移得到下一个状态s_next
            for s_opt in S:
                temp += P.get(join(join(s,a),s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s,a,r,s_next))
            s = s_next
        episodes.append(episode)
    return episodes

def MC(episodes, V, N, gamma):
    '''对所有采样序列计算所有状态的价值'''
    for episode in episodes:
        G = 0
        for i in range(len(episode)-1, -1,-1): #序列要从后往前计算
            (s,a,r,s_next) = episode[i]
            G = r + gamma * G
            N[s] = N[s] + 1
            V[s] = V[s] + (G-V[s]) / N[s]
        
