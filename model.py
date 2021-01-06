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

class SimpleDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(15, 32)
        self.out_w = nn.Linear(32, 1)
        self.out_b = nn.Linear(32, 1)

    def forward(self, x):
        x = self.dense(x)
        out_w = self.out_w(x)
        out_b = self.out_b(x)
        output = out_w + out_b 
        return output 

class SimpleRNN(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.rnn = nn.GRU(32, 32, batch_first=True)
        self.dense_all = nn.Linear(15, 32)
        self.out = nn.Linear(32, 1)
        self.target_length = target_length 

    def forward(self, x):
        x = self.dense_all(x)
        latent, hidden = self.rnn(x)
        output = self.out(latent[:, -self.target_length:])
        return output 

class Seq2Seq(nn.Module):
    def __init__(self, target_length, hidden_size=32):
        super().__init__()
        self.emb = nn.Linear(15, 32)
        self.enc = nn.GRU(32, 32, batch_first=True)
        self.dec = nn.GRU(32, 32, batch_first=True)
        self.att = nn.Linear(hidden_size * 2, 1)
        self.att_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(32, 1)
        self.target_length = target_length 

    def forward(self, x):
        x = self.emb(x)
        latent, hidden = self.enc(x)
        src_len = latent.shape[1]
        enc_hidden = hidden
        if hidden.shape[0] < 2:
            hidden = hidden[0]
        else:
            hidden = hidden[0] + hidden[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        attention = self.att(torch.cat((latent, hidden), dim=-1)).squeeze(-1)
        attention = attention.unsqueeze(1)
        weight = torch.bmm(attention, latent)
        weight = weight.repeat(1, src_len, 1)
        rnn_input = torch.cat((x, weight), dim=-1)
        rnn_input = self.att_combine(rnn_input)
        output, hidden = self.dec(rnn_input, enc_hidden)
        output = self.out(output)
        
        return output 
