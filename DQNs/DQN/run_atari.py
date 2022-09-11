from collections import deque

import torch as T
from dqn import DQNAgent
from utils import device
import numpy as np
from Wrappers.environment import Environment

#Game Setup
env_name = 'PongNoFrameskip-v4'
env = Environment(env_name)
env.seed(544)
test_env = Environment(env_name, episode_life=False, clip_rewards=False)
test_env.seed(644)

train_step = 100000000
log_feq = 1000
test_feq = 100000

def test(agent, step):
    state = test_env.reset()
    done = False
    total_reward = 0
    while not done:
        state_tensor = T.tensor([state], dtype=T.float32, device=device)
        action_tensor = agent.select_action(state_tensor, step)
        next_state, reward, done, _ = test_env.step(action_tensor.item())
        #env.render()
        state = next_state
        total_reward += reward
    #env.close()
    return total_reward

def train():
    agent = DQNAgent(env.get_observation_space().shape[0], env.get_action_space().n, 0, 0.0001)
    agent.train_mode()
    episode_rewards = deque(maxlen=10)

    i_step = 0
    while i_step < train_step:
        state = env.reset()
        done = False
        while not done:
            state_torch = T.tensor([state], dtype=T.float32, device=device)
            action_torch = agent.select_action(state_torch, i_step)

            next_state, reward, done, info = env.step(action_torch.item())
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])

            next_state_torch = T.tensor([next_state], dtype=T.float32) if not done else None
            reward_torch = T.tensor([reward], device=device)
            agent.store_exp(state_torch, action_torch, reward_torch, next_state_torch)

            state = next_state
            agent.learn(i_step, log_feq)

            if i_step % log_feq == 0 and len(episode_rewards) > 1:
                mean_reward = np.mean(episode_rewards)
                agent.calcPerformance(mean_reward, i_step)
                agent.flushTBSummary()

            if i_step % test_feq == 0:
                mean_reward = np.mean([test(agent, i_step) for _ in range(10)])
                print(mean_reward)

            i_step += 1



if __name__ == '__main__':
    train()