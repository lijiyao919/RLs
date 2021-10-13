import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Deep_Q_Network(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims, eta, tb_writer, chkpt_dir='checkpoints'):
        super(Deep_Q_Network, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc3 = nn.Linear(fc1_dims, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=eta)    #0.0001
        # self.optimizer = optim.SGD(self.__policy.parameters(), lr=0.0001)  # 0.01 for method2, 3, online learn
        self.optimizer.zero_grad()

        self.checkpoint_file = os.path.join(chkpt_dir, 'deep_q_network.pth')
        self.writer = tb_writer

    def forward(self, x):
        x = self.fc1(x)
        #x = T.sigmoid(x)
        x = T.tanh(x)
        action_scores = self.fc3(x)
        return action_scores

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def traceWeight(self, epoch):
        self.writer.add_histogram('fc1.weight', self.fc1.weight, epoch)
        self.writer.add_histogram('fc3.weight', self.fc3.weight, epoch)

    def traceBias(self, epoch):
        self.writer.add_histogram('fc1.bias', self.fc1.bias, epoch)
        self.writer.add_histogram('fc3.bias', self.fc3.bias, epoch)

    def traceGrad(self, epoch):
        self.writer.add_histogram('fc1.weight.grad', self.fc1.weight.grad, epoch)
        self.writer.add_histogram('fc1.bias.grad', self.fc1.bias.grad, epoch)
        self.writer.add_histogram('fc3.weight.grad', self.fc3.weight.grad, epoch)
        self.writer.add_histogram('fc3.bias.grad', self.fc3.bias.grad, epoch)