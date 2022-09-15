from collections import deque
import torch as T
from per import PERAgent
from utils import device
import numpy as np
from Wrappers.environment import Environment

#Game Setup
env_name = 'PongNoFrameskip-v4'
env = Environment(env_name)
env.seed(544)

train_step = 100000000
log_feq = 1000

def train():
    agent = PERAgent(env.get_observation_space().shape[0], env.get_action_space().n, 0, 0.0001)
    agent.train_mode()
    episode_rewards = deque(maxlen=10)

    i_step = 0
    while i_step < train_step:
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, i_step)
            next_state, reward, done, info = env.step(action)
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
            agent.store_exp(state, action, reward, next_state, int(done))
            state = next_state
            agent.learn(i_step, log_feq)
            i_step += 1

            if i_step % log_feq == 0 and len(episode_rewards) > 1:
                mean_reward = np.mean(episode_rewards)
                agent.calcPerformance(mean_reward, i_step)
                agent.flushTBSummary()


if __name__ == '__main__':
    train()