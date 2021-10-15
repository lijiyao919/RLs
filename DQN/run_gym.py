import gym
import torch as T
import numpy as np
from dqn import DQNAgent
from itertools import count
from utils import device

env = gym.make('CartPole-v0')
env.seed(0)
TARGET_UPDATE = 10

def train():
    agent = DQNAgent(4, env.action_space.n, 256, 0.001)
    agent.train_mode()

    for i_episode in count(1):
        # Collect from One Episode
        state, ep_reward = env.reset(), 0
        while True:
            state = state.astype(np.float32)
            state_tensor = T.from_numpy(np.expand_dims(state, axis=0)).to(device)
            action_tensor = agent.select_action(state_tensor)
            state, reward, done, info = env.step(action_tensor.item())
            #env.render()

            reward_torch = T.tensor([reward], device=device)
            next_state = state.astype(np.float32)
            next_state_tensor = T.from_numpy(np.expand_dims(next_state, axis=0)).to(device)
            agent.store_exp(state_tensor, action_tensor, reward_torch, next_state_tensor)
            ep_reward += reward
            agent.learn(i_episode, done)
            if done:
                agent.calcPerformance(ep_reward, i_episode)
                agent.flushTBSummary()
                break
    env.close()

if __name__ == '__main__':
    train()