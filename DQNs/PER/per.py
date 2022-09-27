import numpy as np
import random
import math
from statistics import mean
import torch as T
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
from mlp_net import MLP_Network
from cnn_net import CNN_Network
from utils import device

DDQN = True

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done')) #Transition is a class, not object

class PrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = Transition(state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = Transition(*zip(*samples))
        states = np.concatenate(batch.state)
        actions = batch.action
        rewards = batch.reward
        next_states = np.concatenate(batch.next_state)
        dones = batch.done

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

class PERAgent(object):
    def __init__(self, input_dims, n_actions, fc1_dims, eta, buffer_size=100000, batch_size=32, gamma=0.99, target_update_feq=1000, \
                 alpha=0.6, beta_start=0.4, beta_frames = 1000, eps_end=0.1, eps_decay=1000000):
        self.action_space = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update_feq

        self.writer = SummaryWriter()
        self.policy_net = MLP_Network(input_dims, n_actions, fc1_dims, eta, self.writer).to(device)
        #self.policy_net = CNN_Network(input_dims, n_actions, eta, self.writer).to(device)
        self.target_net = MLP_Network(input_dims, n_actions, fc1_dims, eta, self.writer).to(device)
        #self.target_net = CNN_Network(input_dims, n_actions, eta, self.writer).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.eps_start = 1
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        self.replay_buffer = PrioritizedBuffer(buffer_size, self.alpha)

    def store_exp(self, *args):
        self.replay_buffer.push(*args)

    def select_action(self, state, step):
        random_num = random.random()
        eps_thredhold = self.eps_end + (self.eps_start-self.eps_end)*math.exp(-1*step/self.eps_decay)

        if random_num > eps_thredhold:
            state = T.tensor(state, dtype=T.float32, device=device).unsqueeze(0)
            with T.no_grad():
                action = self.policy_net(state).max(1)[1].item()
        else:
            action = random.randrange(self.action_space)
        return action

    def learn(self, step, log_feq):
        beta_by_frame = min(1.0, self.beta_start + step * (1.0 - self.beta_start) / self.beta_frames)
        state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(self.batch_size, beta_by_frame)

        state = T.tensor(state, dtype=T.float32, device=device)
        action = T.tensor(action, dtype=T.long, device=device)
        reward = T.tensor(reward, dtype=T.long, device=device)
        next_state = T.tensor(next_state, dtype=T.float32, device=device)
        done = T.tensor(done, dtype=T.long, device=device)
        weights = T.tensor(weights, dtype=T.float32, device=device)

        q_values = self.policy_net(state)
        next_q_values = self.target_net(next_state)

        if DDQN:
            max_q_act = q_values.max(1)[1]
            next_q_value = next_q_values.gather(1,max_q_act.unsqueeze(1)).squeeze(1)
        else:
            next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        loss = (q_value - expected_q_value.detach()).pow(2) * weights
        prios = loss + 1e-5
        loss = loss.mean()

        self.policy_net.optimizer.zero_grad()
        loss.backward()
        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
        self.policy_net.optimizer.step()

        if step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if step % log_feq == 0:
            self.writer.add_scalar("Loss/train", loss, step)

    def calcPerformance(self, aver_reward, step):
        self.writer.add_scalar("The Average Reward (10 episodes)", aver_reward, step)
        print('Step {}\tThe Average Reward (10 episodes): {:.2f}'.format(step, aver_reward))

    def train_mode(self):
        self.policy_net.train()

    def test_mode(self):
        self.policy_net.eval()

    def flushTBSummary(self):
        self.writer.flush()

