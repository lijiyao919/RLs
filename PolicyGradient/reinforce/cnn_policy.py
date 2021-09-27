import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ReinforceNet(nn.Module):
    def __init__(self, chkpt_dir='checkpoints'):
        super(ReinforceNet, self).__init__()
        self.conve1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=8, stride=4), nn.ReLU())
        self.conve2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(19 * 15 * 32, 256), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256, 2), nn.Softmax(dim=1))

        self.optimizer = optim.RMSprop(self.parameters(), lr=0.0001)
        # self.optimizer = optim.SGD(self.__policy.parameters(), lr=0.0001)  # 0.01 for method2, 3, online learn
        self.optimizer.zero_grad()

        self.checkpoint_file = os.path.join(chkpt_dir, 'reinforce_cnn.pth')

    def forward(self, x):
        x = self.conve1(x)
        x = self.conve2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        action_scores = self.fc2(x)
        return action_scores

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))