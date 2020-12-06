import numpy as np
import torch
import torch.nn as nn 

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(32, 32, batch_first=True)
        self.dense_all = nn.Linear(15, 32)
        self.dense_w = nn.Linear(5, 32)
        self.dense_f = nn.Linear(8, 32)
        self.dense_t = nn.Linear(2, 32)
        self.out_w = nn.Linear(32, 1)
        self.out_b = nn.Linear(32, 1)

    def forward(self, x):
        x_f = self.dense_f(x[:, :, :8])
        x_w = self.dense_w(x[:, :, 8:8+5])
        x_t = self.dense_t(x[:, :, 13:])
        #x = self.dense_all(x)
        latent, hidden = self.rnn(x_f*x_t + x_w*x_t)
        out_w = self.out_w(latent[:, -1])
        out_b = self.out_b(latent[:, -1])
        output = out_w + out_b 
        return output 
