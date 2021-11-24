import gym
import torch as T
import numpy as np
from a2c import A2Cgent
from PolicyGradient.common.utils import device
from Wrappers.venv_wrapper import SubprocVecEnv

env_name = "CartPole-v0"
train_step = 100000
num_step = 5
num_envs = 16

def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env
    return _thunk

def test(agent):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        state_tensor = T.from_numpy(np.expand_dims(state.astype(np.float32), axis=0)).to(device)
        action, _, _,_ = agent.feed_forward(state_tensor)
        next_state, reward, done, _ = env.step(action.item())
        env.render()
        state = next_state
        total_reward += reward
    env.close()
    return total_reward

def train():
    agent = A2Cgent(env.observation_space.shape[0], env.action_space.n, 40, 40, 0.001, num_step, num_envs)
    agent.train_mode()

    i_step = 0
    state = envs.reset()
    while i_step < train_step:
        for _ in range(num_step):
            state_tensor = T.from_numpy(state.astype(np.float32)).to(device)
            action, log_prob, value, entropy = agent.feed_forward(state_tensor) #action:env#, log_prob:env#, value:env#*1 entropt: scale
            next_state, reward, done, _ = envs.step(action.cpu().numpy())  #next_tate: env#*4, reward:env#, done:env#

            value = value.view(num_envs,)
            reward = T.tensor(reward, dtype=T.float32, device=device)
            masks = T.tensor(1-done, dtype=T.long, device=device)
            agent.store_exp(log_prob, value, reward, masks, next_state, entropy)
            state = next_state
            i_step += 1

            if i_step % 1000 == 0:
                mean_reward = np.mean([test(agent) for _ in range(10)])
                agent.calcPerformance(mean_reward, i_step)
                agent.flushTBSummary()

        agent.learn(i_step)


if __name__ == '__main__':
    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)
    env = gym.make(env_name)
    train()