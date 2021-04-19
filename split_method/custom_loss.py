import numpy as np
import torch
from torch import nn

class EXTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = 1

    def forward(self, x, y):
        ret = torch.abs(y-x) + torch.exp((y-x))
        ret = torch.mean(ret)
        return ret
