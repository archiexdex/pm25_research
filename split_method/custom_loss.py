import numpy as np
import torch
from torch import nn

class EXTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = -0.56

    def forward(self, x, y):
        ret = torch.abs(y-x) + torch.exp((y-x))
        #ret = torch.pow(1 + self.shape * torch.abs(x-y), -1/self.shape)
        ret = torch.mean(ret)
        return ret

class EXTActvation(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 1
    
    def forward(self, x):
        # x[x<0] = 0
        # ret = torch.exp(-torch.exp(-x))
        ret = torch.exp(torch.pow(x, self.alpha))
        return ret

class FudanLoss(nn.Module):
    def __init__(self, gamma=1):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, x, y):
        x = nn.Sigmoid()(x)
        normal  = -1 * 0.5 * torch.pow(1-x/self.gamma, self.gamma)     * y * torch.log(x)
        extreme = -1 * 0.5 * torch.pow(1-(1-x)/self.gamma, self.gamma) * (1-y) * torch.log(1-x)
        ret = torch.mean(normal + extreme)
        return ret
