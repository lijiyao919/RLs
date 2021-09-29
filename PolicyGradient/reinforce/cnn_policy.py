import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import device

class CNN_Net(nn.Module):
    def __init__(self, tb_writer, chkpt_dir='checkpoints'):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)      #16*19*19
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=3)     #32*6*6
        self.fc1 = nn.Linear(32*6*6, 256)
        self.fc2 = nn.Linear(256, 2)

        self.optimizer = optim.RMSprop(self.parameters(), lr=0.0001)
        # self.optimizer = optim.SGD(self.__policy.parameters(), lr=0.0001)  # 0.01 for method2, 3, online learn
        self.optimizer.zero_grad()

        self.checkpoint_file = os.path.join(chkpt_dir, 'reinforce_cnn.pth')
        self.writer = tb_writer

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        action_scores = self.fc2(x)
        return F.softmax(action_scores, dim=1)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def traceWeight(self, epoch):
        self.writer.add_histogram('conv1.weight', self.conv1.weight, epoch)
        self.writer.add_histogram('conv2.weight', self.conv2.weight, epoch)
        self.writer.add_histogram('fc1.weight', self.fc1.weight, epoch)
        self.writer.add_histogram('fc2.weight', self.fc2.weight, epoch)

    def traceBias(self, epoch):
        self.writer.add_histogram('conv1.bias', self.conv1.bias, epoch)
        self.writer.add_histogram('conv2.bias', self.conv2.bias, epoch)
        self.writer.add_histogram('fc1.bias', self.fc1.bias, epoch)
        self.writer.add_histogram('fc2.bias', self.fc2.bias, epoch)

    def traceGrad(self, epoch):
        self.writer.add_histogram('conv1.weight.grad', self.conv1.weight.grad, epoch)
        self.writer.add_histogram('conv1.bias.grad', self.conv1.bias.grad, epoch)
        self.writer.add_histogram('conv2.weight.grad', self.conv2.weight.grad, epoch)
        self.writer.add_histogram('conv2.bias.grad', self.conv2.bias.grad, epoch)
        self.writer.add_histogram('fc1.weight.grad', self.fc1.weight.grad, epoch)
        self.writer.add_histogram('fc1.bias.grad', self.fc1.bias.grad, epoch)
        self.writer.add_histogram('fc2.weight.grad', self.fc2.weight.grad, epoch)
        self.writer.add_histogram('fc2.bias.grad', self.fc2.bias.grad, epoch)