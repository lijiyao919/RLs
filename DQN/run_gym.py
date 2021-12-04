import gym
import torch as T
import numpy as np
from dqn import DQNAgent
from utils import device


env_name = "CartPole-v0"
env = gym.make(env_name)
env.seed(0)
test_env = gym.make(env_name)
test_env.seed(10)

train_step = 50000
log_feq = 1000

def test(agent, step):
    state = test_env.reset()
    done = False
    total_reward = 0
    while not done:
        state_tensor = T.from_numpy(np.expand_dims(state.astype(np.float32), axis=0)).to(device)
        action_tensor = agent.select_action(state_tensor, step)
        next_state, reward, done, _ = test_env.step(action_tensor.item())
        #env.render()
        state = next_state
        total_reward += reward
    #env.close()
    return total_reward

def train():
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, 256, 0.001, batch_size=64, target_update_feq=100, eps_end=0.05, eps_decay=10000)
    agent.train_mode()
    i_step = 0

    while i_step < train_step:
        state = env.reset()
        done = False
        while not done:
            state_tensor = T.from_numpy(np.expand_dims(state.astype(np.float32), axis=0)).to(device)
            action_tensor = agent.select_action(state_tensor, i_step)
            state, reward, done, _ = env.step(action_tensor.item())
            # env.render()

            reward_torch = T.tensor([reward], device=device)
            next_state = state.astype(np.float32)
            next_state_tensor = T.from_numpy(np.expand_dims(next_state, axis=0)).to(device)
            agent.store_exp(state_tensor, action_tensor, reward_torch, next_state_tensor)
            agent.learn(i_step, log_feq)
            i_step += 1

            if i_step % log_feq == 0:
                mean_reward = np.mean([test(agent, i_step) for _ in range(10)])
                agent.calcPerformance(mean_reward, i_step)
                agent.flushTBSummary()




if __name__ == '__main__':
    train()