import numpy as np
import random
import torch
import torch.nn as nn 
import torch.nn.functional as F

class Fudan(nn.Module):
    def __init__(self, input_dim, emb_dim, output_dim, hid_dim, device, dropout=0.6, bidirectional=False):
        super().__init__()

        self.bidirectional = bidirectional
        self.emb = nn.Linear(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.hidden_fc = nn.Linear(hid_dim, 1)
        self.out_fc = nn.Linear(hid_dim, 1)
        self.bias_fc = nn.Linear(1, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.device = device

    def forward(self, x, past_window, past_ext):
        # Create a pool to put the predict value
        #batch_size, trg_len, _ = x.shape
        #outputs = torch.zeros(batch_size, trg_len, 1).to(self.device)
        #indicator_outputs = torch.zeros(batch_size, trg_len, 1).to(self.device)
        #hidden = torch.zeros(1, batch_size, history_window[0].shape[-1]).to(self.device)
        #for i in range(trg_len):
        #embed = self.emb(x[:, i:i+1])
        embed = self.emb(x)
        embed = self.dropout(embed)
        latent, hidden = self.rnn(embed)
        #print("latent shape: ", latent.shape)
        #print("hidden shape: ", hidden.shape)
        #hidden = torch.cat((hidden[-1], hidden[-2]), dim=1) if self.bidirectional else hidden[-1]
        # hidden: [1, batch, hid_dim]
        #print("hidden shape: ", hidden.shape)
        
        # Get history window code
        history_window, window_indicator = self.get_window(past_window)
        # attention with window
        alpha = [torch.bmm(hidden.reshape(-1, 1, hidden.shape[-1]), window.reshape(-1, hidden.shape[-1], 1)) for window in history_window]
        alpha = torch.cat(alpha, 1)
        alpha = self.softmax(alpha)
        #print("alpha shape: ", alpha.shape)
        # alpha: [batch, window_len, 1]
        indicator_output = torch.bmm(alpha.reshape(-1, 1, past_ext.shape[1]), past_ext)
        #print("indicator_output shape: ", indicator_output.shape)
        # indicator_output: [batch, 1, 1]
        #output = self.out_fc(latent[:, -1])
        output = self.out_fc(latent[:, -1:])
        bias = self.bias_fc(indicator_output)
        output = output + bias
        # window_indicator: [batch, memory_size, 1]
        # indicator_outputs: [batch, 1, 1]
        # outputs: [batch, 1, 1]
        return window_indicator, indicator_output, output
    
    def get_window(self, past_window):
        # past window
        history_window = []
        for i in range(past_window.shape[1]):
            embed = self.emb(past_window[:, i])
            embed = self.dropout(embed)
            _, hidden = self.rnn(embed)
            hidden = torch.cat((hidden[-1], hidden[-2]), dim=1) if self.bidirectional else hidden[-1]
            hidden = hidden.unsqueeze(1)
            # hidden: [batch, 1, hid_dim]
            history_window.append(hidden)
        window_indicator = torch.cat([self.hidden_fc(window) for window in history_window], 1)
        # window_indicator: [batch, memory_size, 1]
        #print("window_indicator shape: ", window_indicator.shape)
        return history_window, window_indicator

    
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
