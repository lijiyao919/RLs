import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MLP_A2CNet(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims, eta, tb_writer, chkpt_dir='checkpoints'):
        super(MLP_A2CNet, self).__init__()

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=eta)  # 0.0001

        self.checkpoint_file = os.path.join(chkpt_dir, 'mlp_a2c.pth')
        self.writer = tb_writer

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)
        return (F.softmax(pi, dim=1), v)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def traceWeight(self, epoch):
        self.writer.add_histogram('fc1.weight', self.fc1.weight, epoch)
        self.writer.add_histogram('v.weight', self.v.weight, epoch)
        self.writer.add_histogram('pi.weight', self.pi.weight, epoch)

    def traceBias(self, epoch):
        self.writer.add_histogram('fc1.bias', self.fc1.bias, epoch)
        self.writer.add_histogram('v.bias', self.v.bias, epoch)
        self.writer.add_histogram('pi.bias', self.pi.bias, epoch)

    def traceGrad(self, epoch):
        self.writer.add_histogram('fc1.weight.grad', self.fc1.weight.grad, epoch)
        self.writer.add_histogram('fc1.bias.grad', self.fc1.bias.grad, epoch)