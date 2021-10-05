import gym
import torch as T
import numpy as np
from ppo import PPOAgent
from itertools import count
from utils import device

env = gym.make('MountainCar-v0')
env.seed(0)
SEGMENT_SIZE =1024

def train():
    agent = PPOAgent(2, 3, 0.001)
    agent.train_mode()
    step_cnt = 1

    for i_episode in count(1):
        # Collect from One Episode
        state, ep_reward = env.reset(), 0
        while True:
            state = state.astype(np.float32)
            state_tensor = T.from_numpy(np.expand_dims(state, axis=0)).to(device)

            action, log_prob = agent.select_action(state_tensor)
            state, reward, done, _ = env.step(action)
            env.render()
            state = state.astype(np.float32)
            agent.collect_experience(state, action, log_prob, reward)
            ep_reward += reward
            if step_cnt%SEGMENT_SIZE==0:
                agent.learn()
            step_cnt+=1
            if done:
                break
        if ep_reward != -200:
            print("ep "+str(i_episode)+ ": Win!!!!!")
        else:
            print("ep "+str(i_episode)+ ": Fail")
        #agent.calcPerformance(ep_reward, i_episode)
        #agent.flushTBSummary()

if __name__ == '__main__':
    train()