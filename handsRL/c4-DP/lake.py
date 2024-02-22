from cliffwalking import PolicyIteration, ValueIteration,print_agent

import gym
env = gym.make('FrozenLake-v1')
env = env.unwrapped # 解封装以访问状态转移矩阵P
env.render()

holes = set()
ends = set()
for s in env.P:
    for a in env.P[s]:
        for s_ in env.P[s][a]:
            if s_[2] == 1.0 :
                ends.add(s_[1])
            if s_[3] == True:
                holes.add(s_[1])
holes = holes - ends
print('冰洞索引:',holes)
print('目标索引:',ends)

for a in env.P[14]: # 查看左边一格的状态转移信息
    print(env.P[14][a])

# 首先尝试策略梯度
print()
print("-----Beginning of Policy Iteration-----")
print()
action_meaning = ['<','v','>','^']
theta = 1e-5
gamma = 0.9
agent = PolicyIteration(env,theta,gamma)
agent.policy_iteration()
print_agent(agent,action_meaning,[5,7,11,12],[15])
print()
print("-----Ending of Policy Iteration-----")
print()


print()
# 然后是价值迭代
print("-----Beginning of Value Iteration-----")
print()
action_meaning = ['<','v','>','^']
theta = 1e-5
gamma = 0.9
agent = ValueIteration(env,theta,gamma)
agent.value_iteration()
print_agent(agent,action_meaning,[5,7,11,12],[15])

print()
print("-----Ending of Value Iteration-----")
print()