from tkinter import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class NoisyFactorizedLinear(nn.Linear):
    
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.parameter.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        self.register_buffer('epsilon_input', torch.zeros(1, in_features))
        self.register_buffer('epsilon_output', torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.parameter.Parameter(torch.Tensor(out_features).fill_(sigma_init))
    
    def forward(self, input):
        bias = self.bias
        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        
        with torch.no_grad():
            torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
            torch.randn(self.epsilon_output.size(), out=self.epsilon_output)
            eps_in = func(self.epsilon_input)
            eps_out = func(self.epsilon_output)
            noise_v = torch.mul(eps_in, eps_out).detach()
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)


class NoisyNet(ConvBlock):
    
    def __init__(self, n_actions, batch_norm=False):
        super().__init__(batch_norm)
        self.noisy_1 = NoisyFactorizedLinear(3136, 512)
        self.output = NoisyFactorizedLinear(512, n_actions)
    
    def forward(self, x):
        x = super().forward(x)
        x = torch.relu(self.noisy_1(x))
        return self.output(x)

class RainbowNet(ConvBlock):
    def __init__(self, n_actions, batch_norm=False):
        super().__init__(batch_norm)
        self.a_1 = NoisyFactorizedLinear(3136, 512)
        self.a_2 = NoisyFactorizedLinear(512, n_actions)
        self.v_1 = NoisyFactorizedLinear(3136, 512)
        self.v_2 = NoisyFactorizedLinear(512, 1)
    
    def forward(self, x):
        x = super().forward(x)
        a = torch.relu(self.a_1(x))
        a = self.a_2(a)
        
        v = torch.relu(self.v_1(x))
        v = self.v_2(v).expand_as(a)
        
        return v + a - a.mean(1).unsqueeze(1).expand_as(a)