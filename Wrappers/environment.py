import gym
import numpy as np
from Wrappers.atari_wrapper import make_atari, wrap_deepmind
from Wrappers.monitor import Monitor

#Can use for Value based train/test,  and Policy based test
class Environment(object):
    def __init__(self, env_name, episode_life=True, clip_rewards=True, scale=True):
        env = make_atari(env_name)
        env = Monitor(env, allow_early_resets=False)
        self._env = wrap_deepmind(env, episode_life=episode_life, clip_rewards=clip_rewards, frame_stack=True, scale=scale)

    def seed(self, seed):
        self._env.seed(seed)

    def reset(self):
        observation = self._env.reset()
        return np.rollaxis(np.array(observation), 2)   # for pytorch convert (from HWC to CHW)

    def step(self, action):
        if not self._env.action_space.contains(action):
            raise ValueError('Ivalid action!!')
        observation, reward, done, info = self._env.step(action)
        return np.rollaxis(np.array(observation), 2), reward, done, info

    def get_action_space(self):
        return self._env.action_space

    def get_observation_space(self):
        shape = self._env.observation_space.shape
        self._env.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(shape[-1], shape[0], shape[1]), dtype=np.uint8) # for pytorch convert (from HWC to CHW)
        return self._env.observation_space

    def get_random_action(self):
        return self._env.action_space.sample()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()
