import numpy as np
import torch as T
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from mlp_policy import MLP_Net
from statistics import mean
from utils import device

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.log_probs = []
        self.actions = []
        self.rewards = []
        self.batch_size = batch_size

    def generate_index_batches(self):
        n_states = len(self.states)                                        # the len of state should be the same as log_probs, actions and rewards
        batch_start = np.arange(0, n_states, self.batch_size)              # e.g. batch_size=10, then it is [0, 10, 20, 30, ...]
        indices = np.arange(n_states, dtype=np.int64)                      # index of states, reward etc. [0,1,2,3,4,5,6,7,8,9,....]
        batches = [indices[i:i + self.batch_size] for i in batch_start]    # e.g. batch_size=10, then it is [[0,1,2,3,4,5,...,9],[10,...,19],[20,...,29]]
        return np.array(self.states), np.array(self.actions), np.array(self.log_probs), np.array(self.rewards), batches

    def store_memory(self, state, action, log_probs, reward):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_probs)
        self.rewards.append(reward)

    def clear_memory(self):
        del self.states[:]
        del self.log_probs[:]
        del self.actions[:]
        del self.rewards[:]

class PPOAgent(object):
    def __init__(self, input_dims, n_actions, eta, gamma=0.99, epsilon=0.2, batch_size=10000, n_epoch=10):
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_epoch = n_epoch

        self.writer = SummaryWriter()
        self.policy = MLP_Net(input_dims, n_actions, eta, self.writer).to(device)
        self.memo = PPOMemory(batch_size)
        self.recent_rewards = []

    def collect_experience(self, state, action, log_probs, reward):
        self.memo.store_memory(state, action, log_probs, reward)

    def select_action(self,state):
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob.item()

    def learn(self):
        print("I am learning now......")
        state_arr, action_arr, old_prob_arr, reward_arr, batches = self.memo.generate_index_batches()

        #calculate advantage
        r = 0
        returns = []
        for r_pi in reward_arr[::-1]:
            if r_pi == 0:
                r = r_pi + self.gamma * r
            else:
                r = r_pi
            returns.insert(0, r)
        returns = T.tensor(returns, device=device)
        Avantage = (returns - returns.mean()) / returns.std()

        for _ in range(self.n_epoch):
            for batch in batches:
                states_batch = T.from_numpy(state_arr[batch]).to(device)
                old_log_probs_batch = T.tensor(old_prob_arr[batch], device=device)
                actions_batch = T.tensor(action_arr[batch], device=device)

                new_probs_batch = self.policy(states_batch)
                new_probs_batch = Categorical(new_probs_batch)
                new_log_probs_batch = new_probs_batch.log_prob(actions_batch)
                prob_ratios_batch = (new_log_probs_batch - old_log_probs_batch).exp()
                L_CPI = prob_ratios_batch * Avantage[batch]
                L_CLIP = T.clamp(prob_ratios_batch, 1-self.epsilon, 1+self.epsilon) * Avantage[batch]
                loss = - T.min(L_CPI, L_CLIP).sum()
                self.policy.optimizer.zero_grad()
                loss.backward()
                self.policy.optimizer.step()

        self.memo.clear_memory()

    def calcPerformance(self, ep_reward, epoch):
        self.recent_rewards.append(ep_reward)
        if len(self.recent_rewards) > 30:
            self.recent_rewards.pop(0)
        aver_reward = mean(self.recent_rewards)
        if epoch % 10 == 0:
            self.writer.add_scalar("The Average Reward (recent 30 episodes)", aver_reward, epoch)
        print('Episode {}\tReward: {:.2f}\tThe Average Reward (recent 30 episodes): {:.2f}'.format(epoch, ep_reward, aver_reward))

    def flushTBSummary(self):
        self.writer.flush()

    def train_mode(self):
        self.policy.train()

    def test_mode(self):
        self.policy.eval()