import numpy as np
import torch
from torch import nn

EPS = 1e-8

class EVLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, x, y):
        x = nn.Sigmoid()(x)
        normal  = -1 * self.alpha     * torch.pow(1-x    /self.gamma, self.gamma) * y     * torch.log(x   + EPS)
        extreme = -1 * (1-self.alpha) * torch.pow(1-(1-x)/self.gamma, self.gamma) * (1-y) * torch.log(1-x + EPS)
        ret = torch.mean(normal + extreme)
        return ret

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, x, y):
        x = nn.Sigmoid()(x)
        pos = -1 * self.alpha     * torch.pow(1-x    , self.gamma) * y     * torch.log(x   + EPS)
        neg = -1 * (1-self.alpha) * torch.pow(1-(1-x), self.gamma) * (1-y) * torch.log(1-x + EPS)
        ret = torch.mean(pos, neg)
        return ret
