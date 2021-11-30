import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CNN_Network(nn.Module):
    def __init__(self, input_dims, n_actions, eta, tb_writer, chkpt_dir='checkpoints'):
        super(CNN_Network, self).__init__()

        self.conve1 = nn.Conv2d(in_channels=input_dims, out_channels=16, kernel_size=8, stride=4)
        self.conve2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32*9*9, 256)
        self.fc2 = nn.Linear(32*9*9, 256)
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)

        self.optimizer = optim.RMSprop(self.parameters(), lr=eta)  # 0.0001

        self.checkpoint_file = os.path.join(chkpt_dir, 'cnn_dqn.pth')
        self.writer = tb_writer

    def forward(self, x):
        x = x/255.0
        x = F.relu_(self.conve1(x))
        x = F.relu_(self.conve2(x))
        x = x.view(x.size()[0], -1)
        x_pi = F.relu_(self.fc1(x))
        pi = self.pi(x_pi)
        x_v = F.relu_(self.fc2(x))
        v = self.v(x_v)
        return (F.softmax(pi, dim=1), v)


    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    '''def traceWeight(self, epoch):
        self.writer.add_histogram('conve1.weight', self.conve1.weight, epoch)
        self.writer.add_histogram('conve2.weight', self.conve2.weight, epoch)
        self.writer.add_histogram('conve3.weight', self.conve3.weight, epoch)
        self.writer.add_histogram('fc1.weight', self.fc1.weight, epoch)
        self.writer.add_histogram('pi.weight', self.pi.weight, epoch)
        self.writer.add_histogram('v.weight', self.v.weight, epoch)

    def traceBias(self, epoch):
        self.writer.add_histogram('conve1.bias', self.conve1.bias, epoch)
        self.writer.add_histogram('conve2.bias', self.conve2.bias, epoch)
        self.writer.add_histogram('conve3.bias', self.conve3.bias, epoch)
        self.writer.add_histogram('fc1.bias', self.fc1.bias, epoch)
        self.writer.add_histogram('pi.bias', self.pi.bias, epoch)
        self.writer.add_histogram('v.bias', self.v.bias, epoch)

    def traceGrad(self, epoch):
        self.writer.add_histogram('conve1.weight.grad', self.conve1.weight.grad, epoch)
        self.writer.add_histogram('conve2.weight.grad', self.conve2.weight.grad, epoch)
        self.writer.add_histogram('conve3.weight.grad', self.conve3.weight.grad, epoch)
        self.writer.add_histogram('conve1.bias.grad', self.conve1.bias.grad, epoch)
        self.writer.add_histogram('conve2.bias.grad', self.conve2.bias.grad, epoch)
        self.writer.add_histogram('conve3.bias.grad', self.conve3.bias.grad, epoch)
        self.writer.add_histogram('fc1.weight.grad', self.fc1.weight.grad, epoch)
        self.writer.add_histogram('fc1.bias.grad', self.fc1.bias.grad, epoch)
        self.writer.add_histogram('pi.weight.grad', self.pi.weight.grad, epoch)
        self.writer.add_histogram('pi.bias.grad', self.pi.bias.grad, epoch)
        self.writer.add_histogram('v.weight.grad', self.v.weight.grad, epoch)
        self.writer.add_histogram('v.bias.grad', self.v.bias.grad, epoch)'''