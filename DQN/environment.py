import gym
from gym import spaces
import numpy as np
from atari_wrapper import make_wrap_atari


class Environment(object):
    def __init__(self, env_name, clip_rewards=False):
        self._env = make_wrap_atari(env_name, clip_rewards)


    def seed(self, seed):
        self._env.seed(seed)

    def reset(self):
        observation = self._env.reset()
        return np.rollaxis(np.array(observation), 2)

    def step(self, action):
        if not self._env.action_space.contains(action):
            raise ValueError('Ivalid action!!')
        observation, reward, done, info = self._env.step(action)
        return np.rollaxis(np.array(observation), 2), reward, done, info

    def get_action_space(self):
        return self._env.action_space

    def get_observation_space(self):
        shape = self._env.observation_space.shape
        self._env.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(shape[-1], shape[0], shape[1]), dtype=np.uint8)
        return self._env.observation_space

    def get_random_action(self):
        return self._env.action_space.sample()
