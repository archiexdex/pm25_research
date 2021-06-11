import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_loss import *

class GRU(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # Parse paras
        input_dim     = opt.input_dim
        embed_dim     = opt.embed_dim 
        hid_dim       = opt.hid_dim 
        output_dim    = opt.output_dim
        source_size   = opt.source_size
        dropout       = opt.dropout
        num_layers    = opt.num_layers
        bidirectional = opt.bidirectional
        self.target_size = opt.target_size
        # 
        self.dense_emb = nn.Linear(input_dim, embed_dim)
        self.dense_out = nn.Linear(hid_dim*source_size*num_layers, output_dim*self.target_size)
        self.rnn = nn.GRU(embed_dim, hid_dim, batch_first=True, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
    
    def forward(self, x):
        source_size = x.shape[1]
        emb = self.relu(self.dense_emb(x))
        hid, _ = self.rnn(emb)
        hid = self.relu(hid)
        flt = hid.reshape(-1, hid.shape[1] * hid.shape[2])
        flt = self.dropout(flt)
        out = self.relu(self.dense_out(flt))
        out = out.view(-1, self.target_size, 1)
        return emb, hid, flt, out

class DNN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # Parse paras
        input_dim=opt.input_dim
        embed_dim=opt.embed_dim 
        hid_dim=opt.hid_dim
        output_dim=opt.output_dim
        source_size=opt.source_size
        self.target_size=opt.target_size
        # 
        self.dense_emb  = nn.Linear(input_dim, embed_dim)
        self.dense_hid  = nn.Linear(embed_dim, hid_dim)
        self.dense_out  = nn.Linear(hid_dim*source_size, output_dim*self.target_size)
        self.dropout    = nn.Dropout()
        self.relu       = nn.ReLU()
        self.softmax    = nn.Softmax(dim=-1)
        self.leakyrelu  = nn.LeakyReLU()
    
    def forward(self, x):
        source_size = x.shape[1]
        emb = self.relu(self.dense_emb(x))
        hid = self.relu(self.dense_hid(emb))
        flt = hid.view(-1, hid.shape[1] * hid.shape[2])
        flt = self.dropout(flt)
        out = self.relu(self.dense_out(flt))
        out = out.view(-1, self.target_size, 1)
        return emb, hid, flt, out

class DNN_merged(nn.Module):
    def __init__(self, opt, nor_model, ext_model):
        super().__init__()
        # Parse paras
        input_dim=opt.input_dim
        hid_dim = opt.hid_dim
        output_dim=opt.output_dim
        source_size=opt.source_size
        num_layers=opt.num_layers
        self.target_size=opt.target_size
        # 
        self.nor_model = nor_model
        self.ext_model = ext_model
        self.dense_nor = nn.Linear(hid_dim*source_size, hid_dim)
        self.dense_ext = nn.Linear(hid_dim*source_size, hid_dim)
       # self.dense_nor = nn.Linear(hid_dim*source_size*num_layers, hid_dim)
        #self.dense_ext = nn.Linear(hid_dim*source_size*num_layers, hid_dim)
        self.dense_out = nn.Linear(hid_dim*2, output_dim*self.target_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        source_size = x.shape[1]
        nor_emb, nor_hid, nor_flt, nor_out = self.nor_model(x)
        ext_emb, ext_hid, ext_flt, ext_out = self.ext_model(x)
        
        nor_flt = self.relu(self.dense_nor(nor_flt))
        ext_flt = self.relu(self.dense_ext(ext_flt))
        merged_flt = torch.cat((nor_flt, ext_flt), -1)
        out = self.relu(self.dense_out(merged_flt))
        out = out.view(-1, self.target_size, 1)
        return out

class G_DNN(nn.Module):
    def __init__(self, input_dim, embed_dim, hid_dim, output_dim, source_size):
        super().__init__()
        self.dense_emb = nn.Linear(input_dim, embed_dim)
        self.dense_hid = nn.Linear(embed_dim, hid_dim)
        self.dense_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.leakyrelu = nn.LeakyReLU()
    
    def forward(self, x):
        source_size = x.shape[1]
        x = self.relu(self.dense_emb(x))
        x = self.relu(self.dense_hid(x))
        x = self.dense_out(x)[:, -8:]
        return x

class G_GRU(nn.Module):
    def __init__(self, input_dim, embed_dim, hid_dim, output_dim, source_size, dropout, num_layers, bidirectional):
        super().__init__()
        self.dense_emb = nn.Linear(input_dim, embed_dim)
        self.dense_out = nn.Linear(hid_dim*num_layers, output_dim)
        self.rnn = nn.GRU(embed_dim, hid_dim, batch_first=True, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
    
    def forward(self, x):
        source_size = x.shape[1]
        x = self.relu(self.dense_emb(x))
        x, _ = self.rnn(x)
        x = self.relu(x)
        x = self.dense_out(x)[:, -8:]
        return x

class D_DNN(nn.Module):
    def __init__(self, input_dim, embed_dim, hid_dim, output_dim, source_size):
        super().__init__()
        self.dense_emb = nn.Linear(input_dim, embed_dim)
        self.dense_hid = nn.Linear(embed_dim, hid_dim)
        self.dense_out = nn.Linear(hid_dim*source_size, output_dim)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.leakyrelu = nn.LeakyReLU()
    
    def forward(self, x):
        source_size = x.shape[1]
        x = self.relu(self.dense_emb(x))
        x = self.relu(self.dense_hid(x))
        x = self.dense_out(x.view(-1, x.shape[1] * x.shape[2]))
        return x

class D_GRU(nn.Module):
    def __init__(self, input_dim, embed_dim, hid_dim, output_dim, source_size, dropout, num_layers, bidirectional):
        super().__init__()
        self.dense_emb = nn.Linear(input_dim, embed_dim)
        self.dense_out = nn.Linear(hid_dim*source_size*num_layers, output_dim)
        self.rnn = nn.GRU(embed_dim, hid_dim, batch_first=True, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
    
    def forward(self, x):
        source_size = x.shape[1]
        x = self.relu(self.dense_emb(x))
        x, _ = self.rnn(x)
        x = self.relu(x)
        x = x.contiguous()
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = self.dense_out(x)
        return x

class Discrete_DNN(nn.Module):
    def __init__(self, input_dim, embed_dim, hid_dim, output_dim, source_size):
        super().__init__()
        self.dense_emb = nn.Linear(input_dim, embed_dim)
        self.dense_hid = nn.Linear(embed_dim, hid_dim)
        self.dense_out = nn.Linear(hid_dim*source_size, output_dim)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.activation = EXTActvation()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU()
    
    def forward(self, x):
        source_size = x.shape[1]
        x = self.relu(self.dense_emb(x))
        x = self.relu(self.dense_hid(x))
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = self.dense_out(x)
        return x

class Discrete_GRU(nn.Module):
    def __init__(self, input_dim, embed_dim, hid_dim, output_dim, source_size, num_layers, dropout, bidirectional):
        super().__init__()
        self.dense_emb = nn.Linear(input_dim, embed_dim)
        self.rnn = nn.GRU(embed_dim, hid_dim, batch_first=True, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.dense_out = nn.Linear(hid_dim*source_size*num_layers, output_dim)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.activation = EXTActvation()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU()
    
    def forward(self, x):
        source_size = x.shape[1]
        x = self.relu(self.dense_emb(x))
        x, _ = self.rnn(x)
        x = self.relu(x)
        x = x.reshape(-1, x.shape[1] * x.shape[2])
        x = self.dense_out(x)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, hid_dim, dropout=0.6, bidirectional=True):
        super().__init__()
        
        # Parse paras
        input_dim          = opt.input_dim
        embed_dim            = opt.embed_dim 
        hid_dim            = opt.hid_dim 
        output_dim         = opt.output_dim
        dropout            = opt.dropout
        self.bidirectional = opt.bidirectional

        self.emb = nn.Linear(input_dim, embed_dim)
        self.enc = nn.GRU(embed_dim, hid_dim, batch_first=True, bidirectional=bidirectional)
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
    def __init__(self, input_dim, embed_dim, output_dim, hid_dim, attention, dropout=0.6, bidirectional=True):
        super().__init__()
        
        self.attention = attention
        self.emb = nn.Linear(input_dim, embed_dim)
        self.dec = nn.GRU(embed_dim, hid_dim, batch_first=True) if bidirectional else nn.GRU(embed_dim, hid_dim, batch_first=True)
        self.dec_fc = nn.Linear(hid_dim + embed_dim, output_dim) if bidirectional else nn.Linear(hid_dim + embed_dim, output_dim)
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
    def __init__(self, input_dim, embed_dim, output_dim, hid_dim, device, dropout=0.6, bidirectional=True):
        super().__init__()

        input_dim = opt.input_dim
        embed_dim = opt.embed_dim

        self.device = device 
        self.encoder = Encoder(input_dim, embed_dim, output_dim, hid_dim, dropout, bidirectional)
        attention = Attention(hid_dim, bidirectional)
        self.decoder = Decoder(input_dim, embed_dim, output_dim, hid_dim, attention, dropout, bidirectional)

    def forward(self, x, past_window, past_ext, teacher_force_ratio=0.6):
        # Encode
        past_out, past_hidden = self.encoder(past_window)
        # Decode
        predict, y_pred = self.decoder(x, past_hidden, past_out, past_ext)
        
        return predict, y_pred


class Fudan_Encoder(nn.Module):
    def __init__(self, opt, device):
        super().__init__()

        # Parse paras
        input_dim     = opt.input_dim
        embed_dim     = opt.embed_dim 
        hid_dim       = opt.hid_dim 
        dropout       = opt.dropout
        self.bidirectional = False

        self.emb = nn.Linear(input_dim, embed_dim)
        self.rnn = nn.GRU(embed_dim, hid_dim, batch_first=True, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mode):
        if mode == 0:
            # past window
            history_window = [None] * x.shape[1]
            for i in range(x.shape[1]):
                embed = self.emb(x[:, i])
                #embed = self.dropout(embed)
                _, hidden = self.rnn(embed)
                hidden = torch.cat((hidden[-1], hidden[-2]), dim=1) if self.bidirectional else hidden[-1]
                hidden = hidden.unsqueeze(1)
                # hidden: [batch, 1, hid_dim]
                history_window[i] = hidden
            return history_window
        elif mode == 1:
            # current data
            embed = self.emb(x)
            #embed = self.dropout(embed)
            latent, hidden = self.rnn(embed)
            hidden = hidden.reshape(-1, 1, hidden.shape[-1])
            return latent, hidden

class Fudan_History(nn.Module):
    def __init__(self, opt, device):
        super().__init__()

        # Parse paras
        input_dim     = opt.input_dim
        embed_dim     = opt.embed_dim 
        hid_dim       = opt.hid_dim 
        dropout       = opt.dropout
        output_dim    = opt.output_dim
        self.hidden_fc = nn.Linear(hid_dim, output_dim)

    def forward(self, history_window):
        window_indicator = torch.cat([self.hidden_fc(window) for window in history_window], 1)
        # window_indicator: [batch, memory_size, 1]
        return window_indicator

class Fudan_Decoder(nn.Module):
    def __init__(self, opt, device):
        super().__init__()

        # Parse paras
        input_dim     = opt.input_dim
        embed_dim     = opt.embed_dim 
        hid_dim       = opt.hid_dim 
        source_size   = opt.source_size
        dropout       = opt.dropout
        num_layers    = opt.num_layers

        self.memory_size   = opt.memory_size
        self.output_dim    = opt.output_dim
        self.bidirectional = False
        #self.emb = nn.Linear(input_dim, embed_dim)
        #self.rnn = nn.GRU(embed_dim, hid_dim, batch_first=True, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.hidden_fc = nn.Linear(hid_dim, 1)
        self.out_fc = nn.Linear(hid_dim, 1)
        self.bias_fc = nn.Linear(1, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.device = device

    def forward(self, latent, hidden, history_window, past_ext):
        # hidden: [batch, 1, hid_dim]
        # latten: [batch, source_size, hid_dim]
        # attention with window
        #print(hidden.shape, history_window[0].reshape(-1, hidden.shape[-1], 1).shape)
        alpha = [torch.bmm(hidden, window.reshape(-1, hidden.shape[-1], 1)) for window in history_window]
        alpha = torch.cat(alpha, 1)
        alpha = self.softmax(alpha)
        # alpha: [batch, window_len, 1]
        indicator_output = torch.bmm(alpha.reshape(-1, 1, past_ext.shape[1]), past_ext)
        # indicator_output: [batch, 1, 1]
        output = self.out_fc(latent)
        bias = self.bias_fc(indicator_output)
        output = output + bias
        # indicator_output: [batch, 1, 1]
        # output: [batch, 1, 1]
        return output, indicator_output
    

class UNet_1d(nn.Module):
    def __init__(self, c_in, c_hid):
        super().__init__()
        self.dense_0 = nn.Linear(c_in, c_hid)
        self.dense_1 = nn.Linear(c_hid, c_hid * 2)
        self.dense_2 = nn.Linear(c_hid * 2, c_hid * 4)
        self.dense_3 = nn.Linear(c_hid * 4, c_hid * 2)
        self.dense_4 = nn.Linear(c_hid * 4, c_hid * 1)
        self.dense_5 = nn.Linear(c_hid * 2, 1)
        self.relu    = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.dense_0(x))
        x2 = self.relu(self.dense_1(x1))
        x3 = self.relu(self.dense_2(x2))
        x4 = self.relu(self.dense_3(x3))
        x4 = torch.cat((x2, x4), dim=-1)
        x5 = self.relu(self.dense_4(x4))
        x5 = torch.cat((x1, x5), dim=-1)
        x6 = self.relu(self.dense_5(x5))
        return x6
