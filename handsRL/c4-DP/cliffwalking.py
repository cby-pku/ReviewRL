import copy

class CliffWalkingEnv:
    def __init__(self,ncol=12, nrow=4):
        self.ncol = ncol # 定义网格世界的列
        self.nrow = nrow
        # 转移矩阵 P[state][action] = [(p, next_state, reward, done)] 包含下一个状态和奖励
        self.P = self.createP()
    
    def createP(self):
        P = [[[] for j in range(4)] for i in range(self.ncol * self.nrow)]
        # 4种动作, change[0]上,change[1]下,change[2]左,change[3]右. 坐标系原点(0,0)
        # 定义左上角
        change = [[0,-1], [0,1], [-1,0],[1,0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或者目标状态，因为无法继续交互，任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
                        continue
                    # 其他位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x !=self.ncol - 1 : # 下一个位置在悬崖
                            reward = -100
                        
                    P[i*self.ncol + j][a] = [(1,next_state,reward,done)]
        return P


# 策略迭代的实现
class PolicyIteration:
    def __init__(self, env, theta, gamma) -> None:
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow  # 初始化价值为0
        self.pi = [[0.25, 0.25, 0.25, 0.25] for i in range(self.env.ncol * self.env.nrow)] # 初始化为均匀随机策略
        self.theta = theta # 策略评估收敛阈值
        self.gamma = gamma 
    
    def policy_evaluation(self):
        cnt = 1
        while 1 :
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            # 计算所有状态
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = [] # 计算状态s下的所有Q(s,a)值
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done =res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(self.pi[s][a] * qsa) # NOTE Difference between Policy Iteration and Value Interation
                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_v[s]- self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            cnt +=1
        print (f"Policy Evluation ends at round {cnt}")
    
    def policy_improvement(self): # 策略提升
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1-done))
                qsa_list.append(qsa) 
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq) # 计算出所有的最大奖励的动作
            # 这些最大奖励的动作均分概率
            self.pi[s] = [1/cntq if q == maxq else 0 for q in qsa_list]
        print('Finished Policy Improvement !!')
        return self.pi
    def policy_iteration(self):
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improvement()
            if old_pi == new_pi:
                break

def print_agent(agent, action_meaning, disaster = [], end =[]):
    print("状态价值:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()
    
    print("Policy:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态，比如Cliffwalking中的cliff
            if (i * agent.env.ncol + j ) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:
                # 目标状态
                print('EEEE',end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str =''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()

# env = CliffWalkingEnv()
# action_meaning = ['^', 'v','<','>']
# theta = 0.001
# gamma = 0.9
# agent = PolicyIteration(env,theta,gamma)
# agent.policy_iteration()
# print_agent(agent,action_meaning,list(range(37,47)),[47])


# 价值迭代
class ValueIteration:
    def __init__(self,env, theta, gamma) -> None:
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow # 初始化价值为0
        self.theta = theta # 价值收敛阈值
        self.gamma = gamma
        # 价值迭代结束后得到的策略
        self.pi = [None for i in range(self.env.ncol * self.env.nrow)]
    
    def value_iteration(self):
        cnt = 0
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for  s in range(self.env.ncol * self.env.nrow):
                qsa_list = [] # 开始计算状态s下的所有Q(s,a)价值
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done =res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1-done))
                    qsa_list.append(qsa) # 这一行代码和下一行代码是策略迭代和价值迭代的主要区别，策略迭代里还乘了一个概率系数
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))

            self.v = new_v
            if max_diff < self.theta: 
                break
            cnt += 1
        print(f'Value Iteration all for rount {cnt}')
        self.get_policy()

    def get_policy(self): # 根据价值函数导出一个贪婪策略
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state,r, done = res
                    qsa += r + p * self.gamma * self.v[next_state] * (1-done)
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq) # 计算有几个动作得到了最大的Q值
            # 让这些动作均分概率
            self.pi[s] = [1/cntq if q == maxq else 0 for q in qsa_list]


# env = CliffWalkingEnv()
# action_meaning = ['^', 'v','<','>']
# theta = 0.001
# gamma = 0.9
# agent = ValueIteration(env,theta,gamma)
# agent.value_iteration()
# print_agent(agent,action_meaning,list(range(37,47)),[47])