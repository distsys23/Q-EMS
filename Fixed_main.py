from Fixed_env import *
import pickle
import numpy as np


DAY0 = 0
DAYN = 10
REWARDS = {}
for i in range(0,100,1):
    REWARDS[i]=[]

# -------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self,render= False):
        self.env = MicroGridEnv()
        self.render = render
        self.env.day = DAY0
        self.rewards = []
        self.tempt = 0
        # 随机生成240个[0, 35]范围内的随机整数
        self.a = np.random.randint(36, size=240)
        # self.a = np.load("result/baseline2a.npy")
        print(self.a)
    def run(self,day=None):
        s = self.env.reset(day=day)
        R = 0
        t=0
        while True:
            #tcl完全由备用控制决定，价格负载始终不转移优先级始终先从电池走
            #a = random.randint(0, 99)#[0,1,2]4,3,5最后一位能源过剩动作1代表给电池充电,能源不足动作1代表从电池拿电
            # self.a.append(a)
            a=self.a[self.tempt*24+t]
            t=t+1
            s_, r, done, info = self.env.step(a,test=True)
            # print(a)
            if done:  # terminal state
                s_ = None
            s = s_
            R += r
            #print("day="+str(self.env.day)+"time="+str(time)+str(r))
            self.env.render(name='123',display=[0,0,0,0])
            if done:
                # if self.render: self.env.render()
                break
        REWARDS[self.tempt].append(R)
        self.tempt += 1
        self.rewards.append(R)
        print(self.env.day)
        self.env.day+=1
        print("Total reward:", R)

env_test = Environment(render=True)
for day in range(DAY0,DAYN):
    env_test.run(day=day)
print("平均",np.mean(env_test.rewards))
print(np.average([list(REWARDS.values())[DAY0:DAYN]]))
np.save("result/"+'baseline2_profit.npy', env_test.rewards)
print(env_test.rewards)