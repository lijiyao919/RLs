import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import DUELING

class CNN_Network(nn.Module):
    def __init__(self, input_dims, n_actions, eta, tb_writer, chkpt_dir='checkpoints'):
        super(CNN_Network, self).__init__()
        self.conve1 = nn.Conv2d(in_channels=input_dims, out_channels=32, kernel_size=8, stride=4)
        self.conve2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conve3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1_a = nn.Linear(64 * 7 * 7, 512)
        self.fc1_b = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions)
        self.fc3 = nn.Linear(512, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        self.optimizer = optim.RMSprop(self.parameters(), lr=eta)  # 0.0001

        self.checkpoint_file = os.path.join(chkpt_dir, 'cnn_dqn.pth')
        self.writer = tb_writer

    def forward(self, x):
        x = F.relu(self.conve1(x))
        x = F.relu(self.conve2(x))
        x = F.relu(self.conve3(x))
        x = x.contiguous().view(x.size()[0], -1)
        x_a = F.relu(self.fc1_a(x))
        x_b = F.relu(self.fc1_b(x))
        if DUELING:
            V = self.fc3(x_a)
            A = self.fc2(x_b)
            AVER_A = T.mean(A, dim=1, keepdim=True)
            return V+(A-AVER_A)
        else:
            return self.fc2(x)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

