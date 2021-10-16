import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CNN_Network(nn.Module):
    def __init__(self, input_dims, n_actions, eta, tb_writer, chkpt_dir='checkpoints'):
        super(CNN_Network, self).__init__()
        self.conve1 = nn.Conv2d(in_channels=input_dims, out_channels=32, kernel_size=8, stride=4)
        self.conve2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conve3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=eta)  # 0.0001

        self.checkpoint_file = os.path.join(chkpt_dir, 'cnn_dqn.pth')
        self.writer = tb_writer

    def forward(self, x):
        x = F.relu(self.conve1(x))
        x = F.relu(self.conve2(x))
        x = F.relu(self.conve3(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def traceWeight(self, epoch):
        self.writer.add_histogram('conve1.weight', self.conve1.weight, epoch)
        self.writer.add_histogram('conve2.weight', self.conve2.weight, epoch)
        self.writer.add_histogram('conve3.weight', self.conve3.weight, epoch)
        self.writer.add_histogram('fc1.weight', self.fc1.weight, epoch)
        self.writer.add_histogram('fc2.weight', self.fc2.weight, epoch)

    def traceBias(self, epoch):
        self.writer.add_histogram('conve1.bias', self.conve1.bias, epoch)
        self.writer.add_histogram('conve2.bias', self.conve2.bias, epoch)
        self.writer.add_histogram('conve3.bias', self.conve3.bias, epoch)
        self.writer.add_histogram('fc1.bias', self.fc1.bias, epoch)
        self.writer.add_histogram('fc2.bias', self.fc2.bias, epoch)

    def traceGrad(self, epoch):
        self.writer.add_histogram('conve1.weight.grad', self.conve1.weight.grad, epoch)
        self.writer.add_histogram('conve2.weight.grad', self.conve2.weight.grad, epoch)
        self.writer.add_histogram('conve3.weight.grad', self.conve3.weight.grad, epoch)
        self.writer.add_histogram('conve1.bias.grad', self.conve1.bias.grad, epoch)
        self.writer.add_histogram('conve2.bias.grad', self.conve2.bias.grad, epoch)
        self.writer.add_histogram('conve3.bias.grad', self.conve3.bias.grad, epoch)
        self.writer.add_histogram('fc1.weight.grad', self.fc1.weight.grad, epoch)
        self.writer.add_histogram('fc1.bias.grad', self.fc1.bias.grad, epoch)
        self.writer.add_histogram('fc2.weight.grad', self.fc2.weight.grad, epoch)
        self.writer.add_histogram('fc2.bias.grad', self.fc2.bias.grad, epoch)