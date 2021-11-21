from collections import namedtuple
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from mlp_net import MLP_A2CNet
#from cnn_net import CNN_Network
import torch as T
import random
import math
from PolicyGradient.common.utils import device
import torch.nn as nn
from statistics import mean
from torch.distributions import Categorical
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state',  'reward', 'next_state')) #Transition is a class, not object

class RolloutStorage(object):
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
        self.n_plus_1_state = None
        self.total_entropy = 0

    def push(self, log_prob, value, reward, mask, next_state, entropy):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.masks.append(mask)
        self.n_plus_1_state = next_state
        self.total_entropy += entropy
        #print(self.total_entropy)

    def clear(self):
        del self.log_probs[:]
        del self.values[:]
        del self.rewards[:]
        del self.masks[:]
        self.n_plus_1_state = None
        self.total_entropy = 0


class A2Cgent(object):

    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims, eta, n_step, n_env, gamma=0.99):
        self.memo = RolloutStorage()
        self.gamma = gamma
        self.n_step = n_step
        self.n_env = n_env

        self.writer = SummaryWriter()
        self.policy_net = MLP_A2CNet(input_dims, n_actions, fc1_dims, fc2_dims, eta, self.writer).to(device)
        #self.policy_net = CNN_Network(input_dims, n_actions, eta, self.writer).to(device)


    def store_exp(self, state, log_prob, reward, mask, next_state, entropy):
        self.memo.push(state, log_prob, reward, mask, next_state, entropy)


    def feed_forward(self, state):
        prob, value = self.policy_net(state)
        dist = Categorical(prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, dist.entropy().mean()

    def compute_returns(self, n_puls_1_value, rewards, masks):
        R = n_puls_1_value
        returns = []
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * masks[i]
            returns.insert(0, R)
        return returns

    def learn(self, step):
        log_probs_tensor = T.stack(self.memo.log_probs).to(device)
        values_tensor = T.stack(self.memo.values)
        rewards_tensor = T.stack(self.memo.rewards)
        masks_tensor = T.stack(self.memo.masks)
        n_plus_1_state_tensor = T.tensor(self.memo.n_plus_1_state, device=device, dtype=T.float32)

        _, n_plus_1_value = self.policy_net(n_plus_1_state_tensor)
        n_plus_1_value = n_plus_1_value.view(self.n_env, )
        target_values_tensor = self.compute_returns(n_plus_1_value, rewards_tensor, masks_tensor)
        target_values_tensor = T.stack(target_values_tensor)

        advantage = target_values_tensor - values_tensor

        #compute loss
        actor_loss = (-log_probs_tensor*advantage.detach()).mean()
        self.writer.add_scalar("actor loss", actor_loss, step)
        critic_loss = advantage.pow(2).mean()
        self.writer.add_scalar("critic loss", critic_loss, step)
        loss = actor_loss + 0.5*critic_loss - 0.001*self.memo.total_entropy

        self.policy_net.optimizer.zero_grad()
        loss.backward()
        self.policy_net.optimizer.step()

        self.memo.clear()


    def calcPerformance(self, aver_reward, step):
        self.writer.add_scalar("The Average Reward (10 episodes)", aver_reward, step)
        print('In Step: {}\t The Average Reward (10 episodes): {:.2f}'.format(step, aver_reward))


    def flushTBSummary(self):
        self.writer.flush()

    def train_mode(self):
        self.policy_net.train()

    def test_mode(self):
        self.policy_net.eval()

