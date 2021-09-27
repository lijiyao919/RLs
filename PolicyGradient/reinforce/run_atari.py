import cv2
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
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)  # from RGB to Gray scale
    I = I[33:193, 16:144]  # crop the pictuire from 210*160 to 160*130
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

            residual_state = T.from_numpy(residual_state).float().unsqueeze(0).unsqueeze(0)
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