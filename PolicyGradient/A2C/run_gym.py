import gym
import torch as T
import numpy as np
from a2c import A2Cgent
from itertools import count
from utils import device

env = gym.make('CartPole-v0')
env.seed(0)
BATCH_SIZE = 5

def train():
    agent = A2Cgent(env.observation_space.shape[0], env.action_space.n, 40, 40, 0.001, BATCH_SIZE)
    agent.train_mode()
    step = 0

    for i_episode in count(1):
        # Collect from One Episode
        state, ep_reward = env.reset(), 0
        while True:
            step += 1
            state_tensor = T.from_numpy(np.expand_dims(state.astype(np.float32), axis=0)).to(device)
            action, log_prob, entropy = agent.select_action(state_tensor)
            next_state, reward, done, _ = env.step(action)
            #env.render()
            agent.store_exp(state, log_prob, reward, next_state, done, entropy)
            ep_reward += reward
            if agent.store_size() == BATCH_SIZE:
                agent.learn(step)
            if done:
                agent.calcPerformance(ep_reward, i_episode)
                agent.flushTBSummary()
                break
            state = next_state
    env.close()

if __name__ == '__main__':
    train()