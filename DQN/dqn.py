from collections import namedtuple
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from deep_q_network import Deep_Q_Network
import torch as T
import random
import math
from utils import device
import torch.nn as nn
from statistics import mean

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state')) #Transition is a class, not object

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent(object):
    def __init__(self, input_dims, n_actions, fc1_dims, eta, buffer_size=10000, batch_size=32, gamma=0.99, target_update_feq=100):
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.action_space = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update_feq

        self.writer = SummaryWriter()
        self.policy_net = Deep_Q_Network(input_dims, n_actions, fc1_dims, eta, self.writer)
        if T.cuda.device_count() > 1:
            print("Let's use", T.cuda.device_count(), "GPUs!")
            self.policy_net = nn.DataParallel(self.policy_net)
        self.policy_net.to(device)
        self.target_net = Deep_Q_Network(input_dims, n_actions, fc1_dims, eta, self.writer).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.eps_start = 1
        self.eps_end = 0.05
        self.eps_decay = 10000
        self.steps_done = 0

        self.recent_rewards = []

    def store_exp(self, *args):
        self.replay_buffer.push(*args)

    def select_action(self, state):
        random_num = random.random()
        eps_thredhold = self.eps_end + (self.eps_start-self.eps_end)*math.exp(-1*self.steps_done/self.eps_decay)
        self.steps_done +=1

        if random_num > eps_thredhold:
            with T.no_grad():
                return self.policy_net(state).max(1)[1].view(1,1)
        else:
            return T.tensor([[random.randrange(self.action_space)]], device=device)

    def learn(self, epoch, done):
        if len(self.replay_buffer) < self.batch_size:
            return

        #handling samples from
        samples = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*samples))
        state_batch = T.cat(batch.state)
        action_batch = T.cat(batch.action)
        reward_batch = T.cat(batch.reward)

        non_final_mask = T.tensor(tuple(map(lambda x: x is not None, batch.next_state)), device=device, dtype=T.bool)

        non_final_next_state_batch = T.cat([s for s in batch.next_state if s is not None])

        #compute state action values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_action_values = T.zeros(self.batch_size, device=device)
        next_state_action_values[non_final_mask] = self.target_net(non_final_next_state_batch).max(1)[0].detach()
        target_state_action_values = next_state_action_values*self.gamma+reward_batch

        #compute huber loss and optimize
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, target_state_action_values.unsqueeze(1))
        self.writer.add_scalar("Loss/train", loss, epoch)

        self.policy_net.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy_net.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # trance policy parameters
        self.policy_net.traceWeight(epoch)
        self.policy_net.traceBias(epoch)
        self.policy_net.traceGrad(epoch)
        self.record_Q_value(state_action_values, epoch, done)

    def calcPerformance(self, ep_reward, epoch):
        self.recent_rewards.append(ep_reward)
        if len(self.recent_rewards) > 30:
            self.recent_rewards.pop(0)
        aver_reward = mean(self.recent_rewards)
        self.writer.add_scalar("The Episode Reward", ep_reward, epoch)
        self.writer.add_scalar("The Average Reward (recent 30 episodes)", aver_reward, epoch)
        print('Episode {}\tReward: {:.2f}\tThe Average Reward (recent 30 episodes): {:.2f}'.format(epoch, ep_reward, aver_reward))

    def record_Q_value(self, q_values, epoch, done):
        q_arr = []
        mean_q_value = T.mean(T.cat(tuple(q_values.detach()))).item()
        q_arr.append(mean_q_value)
        if done:
            self.writer.add_scalar("The Average Q value", mean(q_arr), epoch)

    def flushTBSummary(self):
        self.writer.flush()

    def train_mode(self):
        self.policy_net.train()

    def test_mode(self):
        self.policy_net.eval()

