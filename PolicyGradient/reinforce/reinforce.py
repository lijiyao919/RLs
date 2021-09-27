import torch as T
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from cnn_policy import ReinforceNet
from statistics import mean

# Device configuration
device = T.device('cuda' if T.cuda.is_available() else 'cpu')
print('The device is: ', device)

class ReinforceMemory(object):
    def __init__(self):
        self.log_probs = []  #for pg
        self.rewards = []

    def clear_memory(self):
        del self.log_probs[:]
        del self.rewards[:]


class ReinforceAgent(object):
    def __init__(self, GAMMA=0.99, mini_batch=10, save_epoch=100, load_policy=False, tb_dir="data"):
        self.gamma = GAMMA
        self.mini_batch = mini_batch
        self.save_epoch = save_epoch

        self.policy = ReinforceNet().to(device)
        self.memo = ReinforceMemory()
        self.writer = SummaryWriter(tb_dir)
        self.recent_rewards = []

        if load_policy:
            self.policy.load_checkpoint()

    def select_action(self, state):
        state = state.to(device)
        probs = self.policy(state)
        # print(probs)
        m = Categorical(probs)
        action = m.sample()
        self.memo.log_probs.append(m.log_prob(action))
        return action.item()

    def learn(self, epoch):
        log_probs = T.cat(self.memo.log_probs).to(device)
        ######################################################
        # method 1:
        # R = T.tensor(self.memo.rewards, device=device).sum()
        # policy_loss = -log_probs*R
        # method 2:
        # R = T.tensor(self.memo.rewards, device=device).sum()
        # policy_loss = -log_probs*(R-Aver_R)
        # method 3:
        r = 0
        returns = []
        for r_pi in self.memo.rewards[::-1]:
            if r_pi == 0:
                r = r_pi + self.gamma * r
            else:
                r = r_pi
            returns.insert(0, r)
        returns = T.tensor(returns, device=device)
        Avantage = (returns - returns.mean()) / returns.std()
        policy_loss = -log_probs * Avantage
        ######################################################
        self.memo.clear_memory()
        policy_loss = policy_loss.sum() / self.mini_batch
        policy_loss.backward()
        if epoch % self.mini_batch == 0:
            self.writer.add_scalar("Loss/train", policy_loss, epoch)
            self.policy.optimizer.step()
            self.policy.optimizer.zero_grad()
        if epoch % self.save_epoch == 0:
            self.policy.save_checkpoint()

    def store_reward(self, reward):
        self.memo.rewards.append(reward)

    def calcPerformance(self, ep_reward, epoch):
        self.recent_rewards.append(ep_reward)
        if len(self.recent_rewards) > 30:
            self.recent_rewards.pop(0)
        aver_reward = mean(self.recent_rewards)
        if epoch % self.mini_batch == 0:
            self.writer.add_scalar("The Average Reward (recent 30 episodes)", aver_reward, epoch)
        print('Episode {}\tReward: {:.2f}\tThe Average Reward (recent 30 episodes): {:.2f}'.format(epoch, ep_reward, aver_reward))

    def flushTBSummary(self):
        self.writer.flush()

    def train_mode(self):
        self.policy.train()

    def test_mode(self):
        self.policy.eval()