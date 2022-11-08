from torch.utils.tensorboard import SummaryWriter
from mlp_net import MLP_Net
from cnn_net import CNN_Network
import torch as T
from utils import device
from torch.distributions import Categorical

class RolloutStorage(object):
    def __init__(self, model):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
        self.target_values_tensor = None
        self.total_entropy = 0
        self.model = model

    def push(self, state, action, log_prob, value, reward, mask, entropy):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.masks.append(mask)
        self.total_entropy += entropy
        #print(self.total_entropy)

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.values[:]
        del self.rewards[:]
        del self.masks[:]
        self.target_values_tensor = None
        self.total_entropy = 0

    def compute_returns(self, n_plus_1_state, n_envs, gamma):
        with T.no_grad():
            _, n_plus_1_value = self.model(n_plus_1_state)
        n_plus_1_value = n_plus_1_value.view(n_envs, )
        R = n_plus_1_value
        returns = []
        for i in reversed(range(len(self.rewards))):
            R = self.rewards[i] + gamma * R * self.masks[i]
            returns.insert(0, R)
        self.target_values_tensor = T.cat(returns)


class PPOAgent(object):
    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims, eta, n_step, n_env, gamma=0.99, ppo_step=4, clip_param=0.2):
        self.gamma = gamma
        self.n_step = n_step
        self.n_env = n_env
        self.ppo_step = ppo_step
        self.clip_param = clip_param

        self.writer = SummaryWriter()
        self.policy_net = MLP_Net(input_dims, n_actions, fc1_dims, fc2_dims, eta, self.writer).to(device)
        #self.policy_net = CNN_Network(input_dims, n_actions, eta, self.writer).to(device)
        self.memo = RolloutStorage(self.policy_net)


    def store_exp(self, state, action, log_prob, value, reward, mask, entropy):
        self.memo.push(state, action, log_prob, value, reward, mask, entropy)

    def compute_target_values(self, n_plus_1_state):
        self.memo.compute_returns(n_plus_1_state, self.n_env, self.gamma)

    def feed_forward(self, state):
        prob, value = self.policy_net(state)
        dist = Categorical(prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, dist.entropy().mean()

    def learn(self):
        state_tensor = T.cat(self.memo.states)
        actions_tensor = T.cat(self.memo.actions)
        old_log_probs_tensor = T.cat(self.memo.log_probs).to(device).detach()
        old_values_tensor = T.cat(self.memo.values).detach()


        advantages = self.memo.target_values_tensor - old_values_tensor

        for _ in range(self.ppo_step):
            new_prob, new_values = self.policy_net(state_tensor)
            dist = Categorical(new_prob)
            new_log_prob = dist.log_prob(actions_tensor)
            new_entropy = dist.entropy().mean()
            new_values = new_values.view(self.n_env*self.n_step, )
            ratio = (new_log_prob-old_log_probs_tensor.detach()).exp()
            surr1 = ratio * advantages
            surr2 = T.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages

            #compute loss
            actor_loss  = - T.min(surr1, surr2).mean()
            critic_loss = (self.memo.target_values_tensor - new_values).pow(2).mean()
            loss = actor_loss + 0.5*critic_loss - 0.001*new_entropy

            self.policy_net.optimizer.zero_grad()
            loss.backward()
            T.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
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

