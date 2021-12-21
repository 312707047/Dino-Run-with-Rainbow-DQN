import torch as T
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        '''input size: (80, 160, 4)'''
        self.conv1 = nn.Conv2d(4, 32, 8, 4, padding='same')
        self.conv2 = nn.Conv2d(32, 64, 4, 2, padding='same')
        self.conv3 = nn.Conv2d(64, 64, 3, 1, padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
    
    def forward(self, x):
        x = T.relu(self.bn1(self.conv1(x)))
        x = T.relu(self.bn2(self.conv2(x)))
        x = T.relu(self.bn3(self.conv3(x)))
        return x.view(x.size(0), -1)

class Network(Conv):
    def __init__(self, n_action):
        self.dense = nn.Linear(3136, 512)
        self.pred = nn.Linear(512, n_action)
    
    def forward(self, x):
        x = super().forward(x)
        x = T.relu(self.dense(x))
        return self.pred(x)