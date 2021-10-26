import gym
import numpy as np
import torch as T
from dqn import DQNAgent
from itertools import count
from utils import device
from atari_wrapper import make_wrap_atari
from environment import Environment

#Game Setup
env = Environment('PongNoFrameskip-v4')
env.seed(544)

def train():
    agent = DQNAgent(env.get_observation_space().shape[0], env.get_action_space().n, 0, 0.0001)
    agent.train_mode()

    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        while True:
            state_torch = T.tensor([state], dtype=T.float32, device=device)
            action_torch = agent.select_action(state_torch)

            next_state, reward, done, _ = env.step(action_torch.item())

            next_state_torch = T.tensor([next_state], dtype=T.float32) if not done else None
            reward_torch = T.tensor([reward], device=device)
            agent.store_exp(state_torch, action_torch, reward_torch, next_state_torch)

            #env.render()
            ep_reward += reward
            if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
                print('ep %d: game finished, reward: %f' % (i_episode, reward) + ('' if reward == -1 else ' !!!!!!!!'))

            state = next_state
            agent.learn(i_episode, done)
            if done:
                agent.calcPerformance(ep_reward, i_episode)
                agent.flushTBSummary()
                break
    env.close()

if __name__ == '__main__':
    train()