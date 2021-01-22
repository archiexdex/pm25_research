import numpy as np
import random
import torch
import torch.nn as nn 
import torch.nn.functional as F

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

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, output_dim, hid_dim, dropout=0.6, bidirectional=True):
        super().__init__()
        
        self.bidirectional = bidirectional
        self.emb = nn.Linear(input_dim, emb_dim)
        self.enc = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.enc_fc = nn.Linear(hid_dim * 2, hid_dim) if bidirectional else nn.Linear(hid_dim, hid_dim)

    def forward(self, x):
        
        embed = self.emb(x)
        embed = self.dropout(embed)
        latent, hidden = self.enc(embed)
        
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1) if self.bidirectional else hidden[-1]
        hidden = self.enc_fc(hidden)
        #hidden = torch.tanh(hidden)
        
        # latent: [batch, src len, hid_dim * num direcions]
        # hidden: [batch, hid_dim]
        return latent, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim, bidirectional=True):
        super().__init__()
        
        self.att = nn.Linear(hid_dim * 2 + hid_dim, hid_dim) if bidirectional else nn.Linear(hid_dim + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, enc_out):
        src_len = enc_out.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # TODO: use torch.cat or plus both
        energy = self.att(torch.cat((enc_out, hidden), dim=-1))
        attention = self.v(energy).squeeze(-1)
        attention = F.softmax(attention, dim=-1)
        
        # attention: [batch size, src len]
        return attention

class Decoder(nn.Module):
    def __init__(self, input_dim, emb_dim, output_dim, hid_dim, attention, dropout=0.6, bidirectional=True):
        super().__init__()
        
        self.attention = attention
        self.emb = nn.Linear(input_dim, emb_dim)
        self.dec = nn.GRU(hid_dim * 2 + emb_dim, hid_dim, batch_first=True) if bidirectional else nn.GRU(hid_dim + emb_dim, hid_dim, batch_first=True)
        self.dec_fc = nn.Linear(hid_dim * 2 + hid_dim + emb_dim, output_dim) if bidirectional else nn.Linear(hid_dim + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, enc_out):
        # x: [batch, input_dim]
        x = x.unsqueeze(1)
        embed = self.emb(x)
        embed = self.dropout(embed)
        a = self.attention(hidden, enc_out)
        a = a.unsqueeze(1)
        weight = torch.bmm(a, enc_out)
        dec_input = torch.cat((embed, weight), dim=-1)
        hidden = hidden.unsqueeze(0)
        latent, hidden = self.dec(dec_input, hidden)
        output = torch.cat((latent, weight, embed), dim=-1)
        predict = self.dec_fc(output)
        return predict, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, emb_dim, output_dim, hid_dim, device, dropout=0.6, bidirectional=True):
        super().__init__()
        
        self.device = device 
        self.encoder = Encoder(input_dim, emb_dim, output_dim, hid_dim, dropout, bidirectional)
        attention = Attention(hid_dim, bidirectional)
        self.decoder = Decoder(output_dim, emb_dim, output_dim, hid_dim, attention, dropout, bidirectional)

    def forward(self, x, y, teacher_force_ratio=0.6):
        
        batch_size = y.shape[0]
        trg_len    = y.shape[1]
        trg_dim    = y.shape[2]

        # Create a pool to put the predict value
        outputs = torch.zeros(batch_size, trg_len, trg_dim).to(self.device)

        # Encode
        enc_out, hidden = self.encoder(x)
        # Decode
        trg = x[: , 0, 7:8]

        for i in range(trg_len):
            output, hidden = self.decoder(trg, hidden, enc_out)
            outputs[:, i] = output[:, 0]
            trg = y[:, i] if random.random() < teacher_force_ratio else output[:, 0]
        
        return outputs

    def interface(self, x):
        
        batch_size = x.shape[0]
        trg_len    = x.shape[1]
        trg_dim    = 1

        # Create a pool to put the predict value
        outputs = torch.zeros(batch_size, trg_len, trg_dim).to(self.device)

        # Encode
        enc_out, hidden = self.encoder(x)
        # Decode
        trg = x[: , 0, 7:8]

        for i in range(trg_len):
            output, hidden = self.decoder(trg, hidden, enc_out)
            outputs[:, i] = output[:, 0]
            trg = output[:, 0]
        
        return outputs
    

