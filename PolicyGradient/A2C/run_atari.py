import torch as T
from a2c import A2Cgent
from utils import device
from collections import deque
from Wrappers.atari_wrapper import make_atari, wrap_deepmind
from Wrappers.venv_wrapper import SubprocVecEnv, VecPyTorchFrameStack, VecPyTorch, TransposeImage
from Wrappers.monitor import Monitor
from Wrappers.environment import Environment
import numpy as np

#Game Setup
num_step = 5
num_envs = 16
train_step = 1000000
log_feq = 1000
test_feq = 10000
env_name = 'PongNoFrameskip-v4'


def make_env(seed, rank):
    def _thunk():
        env = make_atari(env_name)
        env.seed(seed+rank)
        env = Monitor(env, allow_early_resets=False)
        env = wrap_deepmind(env, scale=True)
        env = TransposeImage(env, op=[2, 0, 1])
        return env
    return _thunk

def test(agent):
    state = test_env.reset()
    done = False
    while not done:
        state_tensor = T.from_numpy(np.expand_dims(state.astype(np.float32), axis=0)).to(device)
        action, _, _,_ = agent.feed_forward(state_tensor)
        next_state, reward, done, info = test_env.step(action.item())
        test_env.render()
        state = next_state
        if done:
            test_env.close()
            return info['episode']['r']

def train():
    agent = A2Cgent(envs.observation_space.shape[0], envs.action_space.n, 40, 40, 1e-4, num_step, num_envs)
    agent.train_mode()
    episode_rewards = deque(maxlen=10)

    i_step = 0
    state = envs.reset()
    while i_step < train_step:
        for _ in range(num_step):
            s = state.clone()
            action, log_prob, value, entropy = agent.feed_forward(s)
            next_state, reward, done, infos = envs.step(action)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            reward1 = reward.squeeze(1).to(device)
            value1 = value.squeeze(1).to(device)
            masks1 = T.tensor(1 - done, dtype=T.long, device=device)
            agent.store_exp(log_prob, value1, reward1, masks1, next_state, entropy)
            state = next_state
            i_step += 1

            if i_step % log_feq == 0 and len(episode_rewards) > 1:
                mean_reward = np.mean(episode_rewards)
                agent.calcPerformance(mean_reward, i_step)
                agent.flushTBSummary()

            if i_step % test_feq == 0:
                mean_reward = np.mean([test(agent) for _ in range(10)])
                print(mean_reward)

        agent.learn(i_step, log_feq)


if __name__ == '__main__':
    envs = [make_env(100, i) for i in range(num_envs)]
    envs = SubprocVecEnv(envs)
    envs = VecPyTorch(envs, device)
    envs = VecPyTorchFrameStack(envs, 4, device)

    test_env = Environment(env_name, episode_life=False, clip_rewards=False, scale=True)
    train()