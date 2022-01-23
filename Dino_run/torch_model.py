import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    
    def __init__(self, batch_norm=False):
        super().__init__()
        
        if not batch_norm:
            self.ConvBlock = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
        
        else:
            self.ConvBlock = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.BatchNorm2d(64)
            )
    
    def forward(self, x):
        x = self.ConvBlock(x)
        x = nn.Flatten()(x)
        return x


class Net(ConvBlock):
    
    def __init__(self, n_actions, batch_norm=False):
        super().__init__(batch_norm)
        self.dense = nn.Linear(3136, 512)
        self.output = nn.Linear(512, n_actions)
    
    def forward(self, x):
        x = super().forward(x)
        x = torch.relu(self.dense(x))
        return self.output(x)


class DuelNet(ConvBlock):
    
    def __init__(self, n_actions, batch_norm=False):
        super().__init__(batch_norm)
        self.a_1 = nn.Linear(3136, 512)
        self.a_2 = nn.Linear(512, n_actions)
        self.v_1 = nn.Linear(3136, 512)
        self.v_2 = nn.Linear(512, 1)
    
    def forward(self, x):
        x = super().forward(x)
        a = torch.relu(self.a_1(x))
        a = self.a_2(a)
        
        v = torch.relu(self.v_1(x))
        v = self.v_2(v).expand_as(a)
        
        return v + a - a.mean(1).unsqueeze(1).expand_as(a)