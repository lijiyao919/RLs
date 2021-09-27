import gym
import torch as T
from itertools import count

from reinforce import ReinforceAgent

#Game Setup
env = gym.make('Pong-v0')
env.seed(543)
T.manual_seed(543)
UP = 2
DOWN = 5



def prepro(I):
    """ using Karpathy's code, prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I

def train():
    agent = ReinforceAgent()
    agent.train_mode()
    prev_state = None

    for i_episode in count(1):
        # Collect from One Episode
        state, ep_reward = env.reset(), 0
        while True:
            # preprocess the observation, set input to network to be difference image
            cur_state = prepro(state)
            residual_state = cur_state - prev_state if prev_state is not None else cur_state
            prev_state = cur_state

            action = agent.select_action(residual_state)
            if action == 0:
                state, reward, done, _ = env.step(UP)
            else:
                state, reward, done, _ = env.step(DOWN)
            # env.render()
            agent.store_reward(reward)
            ep_reward += reward
            if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
                print('ep %d: game finished, reward: %f' % (i_episode, reward) + ('' if reward == -1 else ' !!!!!!!!'))
            if done:
                break
        agent.calcPerformance(ep_reward, i_episode)
        agent.learn(i_episode)
        agent.flushTBSummary()

if __name__ == '__main__':
    train()