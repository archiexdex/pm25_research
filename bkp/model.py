import numpy as np
import random
import torch
import torch.nn as nn 
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_dim, emb_dim, output_dim, seq_len, trg_len, device, dropout=0.6):
        super().__init__()
        self.emb = nn.Conv1d(input_dim, emb_dim, 3, padding=1)
        self.net = nn.Sequential(*[
                        nn.Conv1d(emb_dim, emb_dim, 3, padding=1),
                        nn.BatchNorm1d(emb_dim),
                        #nn.InstanceNorm1d(emb_dim),
                        nn.Tanh()
                    ])
        self.out = nn.Conv1d(emb_dim, output_dim, 3, padding=1)
        self.fc  = nn.Linear(seq_len, trg_len)


    def forward(self, x, past_window, past_ext):
        # x: [batch, channel, length]
        # past_window: [batch, channel, length]
        # past_ext: [batch, channel, length]
        x = torch.cat((past_window, x), -1)
        x = self.emb(x)
        x = self.net(x)
        x = self.out(x)
        x = self.fc(x)
        return x

class CNN2DModel(nn.Module):
    def __init__(self, input_dim, emb_dim, output_dim, seq_len, trg_len, device, dropout=0.6):
        super().__init__()
        self.emb = nn.Conv2d(input_dim, emb_dim, 3, padding=1)
        self.net = nn.Sequential(*[
                        nn.Conv2d(emb_dim, emb_dim, 3, padding=1),
                        nn.BatchNorm2d(emb_dim),
                        #nn.InstanceNorm1d(emb_dim),
                        nn.Tanh()
                    ])
        self.out = nn.Conv2d(emb_dim, output_dim, 3, padding=1)
        self.fc  = nn.Linear(seq_len, trg_len)


    def forward(self, x, past_window, past_ext):
        # x: [batch, channel, length]
        # past_window: [batch, channel, length]
        # past_ext: [batch, channel, length]
        x = torch.cat((past_window, x), -1)
        x = self.emb(x)
        x = self.net(x)
        x = self.out(x)
        x = self.fc(x)
        return x
class UNET(nn.Module):
    def __init__(self, input_dim, emb_dim, output_dim, seq_len, trg_len, device, dropout=0.6):
        super().__init__()

        def _conv(in_c, out_c, kernel=3, stride=1):
            return nn.Sequential(*[
                    nn.Conv1d(in_c, out_c, kernel_size=kernel, stride=stride, padding=kernel//2),
                    #nn.BatchNorm1d(out_c),
                    nn.InstanceNorm1d(out_c),
                    nn.Tanh()
            ])
        
        self.emb = nn.Conv1d(input_dim, emb_dim, 3, padding=1)
        self.conv1 = _conv(emb_dim * 1, emb_dim * 2)
        self.conv2 = _conv(emb_dim * 2, emb_dim * 4)
        self.conv3 = _conv(emb_dim * 4, emb_dim * 4)
        self.deconv2 = _conv(emb_dim * 8, emb_dim * 2)
        self.deconv1 = _conv(emb_dim * 4, emb_dim * 1)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.out = nn.Conv1d(emb_dim, output_dim, 3, padding=1)
        self.fc  = nn.Linear(seq_len, trg_len)


    def forward(self, x, past_window, past_ext):
        # x: [batch, channel, length]
        # past_window: [batch, channel, length]
        # past_ext: [batch, channel, length]
        x = torch.cat((past_window, x), -1)
        x = self.emb(x)
        x1 = self.conv1(x)
        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)
        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)
        x4 = self.upsample(x3)
        x4 = torch.cat((x2, x4), 1)
        x4 = self.deconv2(x4)
        x5 = self.upsample(x4)
        x5 = torch.cat((x1, x5), 1)
        x5 = self.deconv1(x5)
        out = self.out(x5)
        out = self.fc(out)
        return out

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
        embed = self.emb(x)
        embed = self.dropout(embed)
        latent, hidden = self.rnn(embed)
        # hidden: [1, batch, hid_dim]
        
        # Get history window code
        history_window, window_indicator = self.get_window(past_window)
        # attention with window
        alpha = [torch.bmm(hidden.reshape(-1, 1, hidden.shape[-1]), window.reshape(-1, hidden.shape[-1], 1)) for window in history_window]
        alpha = torch.cat(alpha, 1)
        alpha = self.softmax(alpha)
        # alpha: [batch, window_len, 1]
        indicator_output = torch.bmm(alpha.reshape(-1, 1, past_ext.shape[1]), past_ext)
        # indicator_output: [batch, 1, 1]
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
        self.v = nn.Linear(hid_dim, hid_dim, bias=False)

    def forward(self, hidden, enc_out):
        src_len = enc_out.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # TODO: use torch.cat or plus both
        energy = self.att(torch.cat((enc_out, hidden), dim=-1))
        attention = self.v(energy)
        attention = F.softmax(attention, dim=-1)
        
        # attention: [batch size, src len]
        return attention

class Decoder(nn.Module):
    def __init__(self, input_dim, emb_dim, output_dim, hid_dim, attention, dropout=0.6, bidirectional=True):
        super().__init__()
        
        self.attention = attention
        self.emb = nn.Linear(input_dim, emb_dim)
        self.dec = nn.GRU(emb_dim, hid_dim, batch_first=True) if bidirectional else nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.dec_fc = nn.Linear(hid_dim + emb_dim, output_dim) if bidirectional else nn.Linear(hid_dim + emb_dim, output_dim)
        self.att_fc = nn.Linear(hid_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.bidirectional = bidirectional

    def forward(self, x, past_hidden, past_out, past_ext):
        embed = self.emb(x)
        embed = self.dropout(embed)
        latent, hidden = self.dec(embed)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1) if self.bidirectional else hidden[-1]
        #print(latent.shape, hidden.shape, past_out.shape)
        a = self.attention(hidden, past_out)
        #print("a ", a.shape)
        #print("past_ext ", past_ext.shape)
        weight = torch.bmm(past_ext.reshape(-1, past_ext.shape[2], past_ext.shape[1]), a)
        #print(weight.shape, latent.shape)
        output = torch.cat((latent[:, -1:], weight), dim=-1)
        #print(output.shape)
        #hidden = hidden.unsqueeze(0)
        #latent, hidden = self.dec(dec_input, hidden)
        #output = torch.cat((latent, weight, embed), dim=-1)
        y_pred = self.att_fc(weight)
        predict = self.dec_fc(output)
        #print(predict.shape, y_pred.shape)
        return predict, y_pred #hidden.squeeze() #hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, emb_dim, output_dim, hid_dim, device, dropout=0.6, bidirectional=True):
        super().__init__()
        self.device = device 
        self.encoder = Encoder(input_dim, emb_dim, output_dim, hid_dim, dropout, bidirectional)
        attention = Attention(hid_dim, bidirectional)
        self.decoder = Decoder(input_dim, emb_dim, output_dim, hid_dim, attention, dropout, bidirectional)

    def forward(self, x, past_window, past_ext, teacher_force_ratio=0.6):
        # Encode
        past_out, past_hidden = self.encoder(past_window)
        # Decode
        predict, y_pred = self.decoder(x, past_hidden, past_out, past_ext)
        
        return predict, y_pred

class SimpleLSTM2d(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, target_length):
        super().__init__()
        self.dense_all = nn.Linear(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.out = nn.Linear(hid_dim, output_dim)
        self.target_length = target_length 

    def forward(self, x, past_window, past_ext):
        x = torch.cat((past_window, x), dim=-1)
        x = self.dense_all(x)
        latent, hidden = self.rnn(x)
        output = self.out(latent[:, :, -1:])
        return output 

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, target_length):
        super().__init__()
        self.dense_all = nn.Linear(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.out = nn.Linear(hid_dim, output_dim)
        self.target_length = target_length 

    def forward(self, x, past_window, past_ext):
        x = torch.cat((past_window, x), dim=1)
        x = self.dense_all(x)
        latent, hidden = self.rnn(x)
        output = self.out(latent[:, -1])
        return output 

class SimpleGRU(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, target_length):
        super().__init__()
        self.dense_all = nn.Linear(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.out = nn.Linear(hid_dim, output_dim)
        self.target_length = target_length 

    def forward(self, x, past_window, past_ext):
        x = torch.cat((past_window, x), dim=1)
        x = self.dense_all(x)
        latent, hidden = self.rnn(x)
        output = self.out(latent[:, -1])
        return output 

class SimpleDNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, target_length):
        super().__init__()
        self.dense_emb = nn.Linear(input_dim, emb_dim)
        self.dense_hid = nn.Linear(emb_dim, hid_dim)
        self.dense_out = nn.Linear(hid_dim, output_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x, past_window, past_ext):
        x = torch.cat((past_window, x), dim=1)
        x = self.relu(self.dense_emb(x))
        x = self.relu(self.dense_hid(x))
        x = self.relu(self.dense_out(x))[:, -1]
        return x 
