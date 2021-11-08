from collections import namedtuple
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from mlp_net import MLP_A2CNet
#from cnn_net import CNN_Network
import torch as T
import random
import math
from utils import device
import torch.nn as nn
from statistics import mean
from torch.distributions import Categorical
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state',  'reward', 'next_state')) #Transition is a class, not object

class A2CMemory(object):
    def __init__(self):
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.last_state = None

    def store_memory(self, state, log_prob, reward, next_state, done):
        self.states.append(state)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.last_state = next_state

    def clear(self):
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.dones[:]
        self.last_state = None

    def __len__(self):
        return len(self.dones)

class A2Cgent(object):

    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims, eta, batch_size, gamma=0.99):
        self.memo = A2CMemory()
        self.gamma = gamma
        self.batch_size = batch_size

        self.writer = SummaryWriter()
        self.policy_net = MLP_A2CNet(input_dims, n_actions, fc1_dims, fc2_dims, eta, self.writer).to(device)
        #self.policy_net = CNN_Network(input_dims, n_actions, eta, self.writer).to(device)

        self.recent_rewards = []
        self.q_arr = []

    def store_exp(self, state, log_prob, reward, next_state, done):
        self.memo.store_memory(state, log_prob, reward, next_state, done)

    def store_size(self):
        return len(self.memo)

    def select_action(self, state):
        probs, _ = self.policy_net(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_probs = dist.log_prob(action)
        return action.item(), log_probs

    def learn(self, step):
        state_tensor = T.tensor(self.memo.states, device=device, dtype=T.float32)
        log_prob_tensor = T.cat(self.memo.log_probs).to(device)
        reward_tensor = T.tensor(self.memo.rewards,device=device)
        done_tensor = T.tensor(self.memo.dones, device=device, dtype=T.uint8)
        last_state_tensor = T.tensor([self.memo.last_state], device=device, dtype=T.float32)

        R = T.zeros(self.batch_size, device=device)
        _, R[-1] = self.policy_net(last_state_tensor)
        for i in range(self.batch_size-1, 0, -1):
            R[i-1] = reward_tensor[i-1]+self.gamma*R[i]*(1-done_tensor[i].item())
        #print(R)

        #compute state action values
        _, V = self.policy_net(state_tensor)
        V = V.view(self.batch_size)
        #print(V)


        #compute loss
        self.writer.add_scalar("V", V.mean(), step)
        advantage = R - V
        #print(advantage)
        #print(log_prob_tensor)
        #print(self.log_prob)
        actor_loss = (-log_prob_tensor*advantage).mean()
        self.writer.add_scalar("actor loss", actor_loss, step)
        critic_loss = advantage.pow(2).mean()
        self.writer.add_scalar("critic loss", critic_loss, step)
        loss = actor_loss + 0.5*critic_loss

        self.policy_net.optimizer.zero_grad()
        loss.backward()
        self.policy_net.optimizer.step()

        self.memo.clear()


    def calcPerformance(self, ep_reward, epoch):
        self.recent_rewards.append(ep_reward)
        if len(self.recent_rewards) > 30:
            self.recent_rewards.pop(0)
        aver_reward = mean(self.recent_rewards)
        self.writer.add_scalar("The Episode Reward", ep_reward, epoch)
        self.writer.add_scalar("The Average Reward (recent 30 episodes)", aver_reward, epoch)
        print('Episode {}\tReward: {:.2f}\tThe Average Reward (recent 30 episodes): {:.2f}'.format(epoch, ep_reward, aver_reward))

    '''def record_Q_value(self, q_values, epoch, done):
        mean_q_value = T.mean(T.cat(tuple(q_values.detach()))).item()
        self.q_arr.append(mean_q_value)
        if done:
            self.writer.add_scalar("The Average Q value", mean(self.q_arr), epoch)
            del self.q_arr[:]'''

    def flushTBSummary(self):
        self.writer.flush()

    def train_mode(self):
        self.policy_net.train()

    def test_mode(self):
        self.policy_net.eval()

