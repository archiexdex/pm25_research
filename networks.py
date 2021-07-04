import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_loss import *
import random

class GRU(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # Parse paras
        input_dim     = opt.input_dim
        embed_dim     = opt.embed_dim 
        hid_dim       = opt.hid_dim 
        output_dim    = opt.output_dim
        source_size   = opt.source_size
        memory_size   = opt.memory_size
        dropout       = opt.dropout
        num_layers    = opt.num_layers
        bidirectional = opt.bidirectional
        self.target_size = opt.target_size
        # 
        self.dense_emb = nn.Linear(input_dim, embed_dim)
        self.dense_out = nn.Linear(hid_dim*num_layers*(memory_size + source_size), output_dim*self.target_size)
        self.rnn = nn.GRU(embed_dim, hid_dim, batch_first=True, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
    
    def forward(self, x, past_window):
        x = torch.cat((past_window, x), dim=1)
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
        memory_size = opt.memory_size
        self.target_size=opt.target_size
        # 
        self.dense_emb  = nn.Linear(input_dim, embed_dim)
        self.dense_hid  = nn.Linear(embed_dim, hid_dim)
        self.dense_out  = nn.Linear(hid_dim*(memory_size + source_size), output_dim*self.target_size)
        self.dropout    = nn.Dropout()
        self.relu       = nn.ReLU()
        self.softmax    = nn.Softmax(dim=-1)
        self.leakyrelu  = nn.LeakyReLU()
    
    def forward(self, x, past_window):
        x = torch.cat((past_window, x), dim=1)
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
        input_dim          = input_dim
        embed_dim          = embed_dim 
        hid_dim            = hid_dim 
        output_dim         = output_dim
        dropout            = dropout
        self.bidirectional = bidirectional

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
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, enc_out):
        src_len = enc_out.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # TODO: use torch.cat or plus both
        energy = self.att(torch.cat((enc_out, hidden), dim=-1))
        a = self.v(energy).squeeze()
        a = F.softmax(a, dim=-1)
        
        # a: [batch size, src len]
        return a

class Decoder(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, hid_dim, attention, dropout=0.6, bidirectional=True):
        super().__init__()
        
        self.bidirectional = bidirectional
        
        self.attention = attention
        self.emb = nn.Linear(input_dim, embed_dim)
        self.dec = nn.GRU(embed_dim + hid_dim * 2, hid_dim, batch_first=True) if bidirectional else nn.GRU(embed_dim + hid_dim, hid_dim, batch_first=True)
        self.dec_fc = nn.Linear(hid_dim, output_dim)
        self.out_fc = nn.Linear(hid_dim*2 + hid_dim + embed_dim, output_dim) if bidirectional else nn.Linear(hid_dim + hid_dim + embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, enc_out):
        embed = self.emb(x)
        embed = self.dropout(embed).unsqueeze(1)
        #print("embed: ", embed.shape)
        #print("hidden:", hidden.shape)
        #print("enc_out: ", enc_out.shape)
        a = self.attention(hidden, enc_out)
        a = a.unsqueeze(1)
        #print("a ", a.shape)
        weight = torch.bmm(a, enc_out)
        #print("weight: ",  weight.shape)
        rnn_input = torch.cat((embed, weight), dim=-1)
        #print("rnn_input: ", rnn_input.shape)
        output, hidden = self.dec(rnn_input, hidden.unsqueeze(0))
        hidden = hidden.squeeze(0)
        #print("output: ", output.shape)
        #print("hidden:", hidden.shape)
        prediction = self.out_fc(torch.cat((output, weight, embed), dim=-1))
        output = self.dec_fc(output)
        #print("output: ", output.shape)
        #print("hidden:", hidden.shape)
        #print("prediction:", prediction.shape)
        return output.squeeze(-1), hidden, prediction.squeeze(-1)


class Seq2Seq(nn.Module):
    def __init__(self, opt, device):
        super().__init__()

        input_dim  = opt.input_dim
        embed_dim  = opt.embed_dim
        hid_dim    = opt.hid_dim
        dropout    = opt.dropout
        bidirectional = opt.bidirectional
        self.output_dim  = opt.output_dim
        self.target_size = opt.target_size
        self.device      = device
        # Model
        self.encoder = Encoder(input_dim, embed_dim, self.output_dim, hid_dim, dropout, bidirectional)
        attention    = Attention(hid_dim, bidirectional)
        self.decoder = Decoder(input_dim, embed_dim, self.output_dim, hid_dim, attention, dropout, bidirectional)

    def forward(self, xs, past_window, teacher_force_ratio=0.6):
        batch_size  = xs.shape[0]
        trg_size    = self.target_size
        output_size = self.output_dim
        # Tensor to store decoder outputs
        outputs     = torch.zeros(batch_size, trg_size, output_size).to(self.device)
        predictions = torch.zeros(batch_size, trg_size, output_size).to(self.device)
        # Encode
        enc_out, hidden = self.encoder(past_window)
        x = xs[:, 0]
        for i in range(trg_size):
            # Decode
            output, hidden, prediction = self.decoder(x, hidden, enc_out)
            # place predictions in a tensor holding predictions for each token
            outputs    [:, i] = output
            predictions[:, i] = prediction
            # teacher force
            #teacher_force = random.random() < teacher_force_ratio
            x = xs[:, i] # if teacher_force else output
        
        return outputs, predictions 


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
        self.softmax = nn.Softmax(dim=1)
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

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, opt, device):
        super().__init__()

        self.hid_dim = opt.hid_dim
        self.n_heads = opt.n_heads
        dropout = opt.dropout

        assert self.hid_dim % self.n_heads == 0
        self.head_dim = self.hid_dim // self.n_heads

        self.fc_q = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc_k = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc_v = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc_o = nn.Linear(self.hid_dim, self.hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value):
        #query = [batch size, query len, hid dim]
        #key   = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]

        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, self.n_heads, -1, self.head_dim)
        K = K.view(batch_size, self.n_heads, -1, self.head_dim)
        V = V.view(batch_size, self.n_heads, -1, self.head_dim)
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        #energy = [batch size, n heads, query len, key len]

        attention = torch.softmax(energy, dim = -1)
        #attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)
        #x = [batch size, n heads, query len, head dim]

        x = x.contiguous().view(batch_size, -1, self.hid_dim)
        #x = [batch size, query len, hid dim]

        x = self.fc_o(x)
        #x = [batch size, query len, hid dim]

        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, opt):
        super().__init__()

        hid_dim = opt.hid_dim
        pf_dim  = opt.pf_dim
        dropout = opt.dropout
        
        self.fc_1    = nn.Linear(hid_dim, pf_dim)
        self.fc_2    = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #x = [batch size, seq len, hid dim]

        x = self.dropout(F.relu(self.fc_1(x)))
        #x = [batch size, seq len, pf dim]

        x = self.fc_2(x)
        #x = [batch size, seq len, hid dim]

        return x

class EncoderLayer(nn.Module):
    def __init__(self, opt, device):
        super().__init__()

        hid_dim = opt.hid_dim
        n_heads = opt.n_heads
        pf_dim  = opt.pf_dim
        dropout = opt.dropout

        self.self_attn_layer_norm     = nn.LayerNorm(hid_dim)
        self.ff_layer_norm            = nn.LayerNorm(hid_dim)
        self.self_attention           = MultiHeadAttentionLayer(opt, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(opt)
        self.dropout                  = nn.Dropout(dropout)

    def forward(self, src):
        #src = [batch size, src len, hid dim]

        #self attention
        _src, _ = self.self_attention(src, src, src)

        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        #src = [batch size, src len, hid dim]

        #positionwise feedforward
        _src = self.positionwise_feedforward(src)

        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        #src = [batch size, src len, hid dim]

        return src
class DecoderLayer(nn.Module):
    def __init__(self, opt, device):
        super().__init__()

        hid_dim = opt.hid_dim
        n_heads = opt.n_heads
        pf_dim  = opt.pf_dim 
        dropout = opt.dropout

        self.self_attn_layer_norm     = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm      = nn.LayerNorm(hid_dim)
        self.ff_layer_norm            = nn.LayerNorm(hid_dim)
        self.self_attention           = MultiHeadAttentionLayer(opt, device)
        self.encoder_attention        = MultiHeadAttentionLayer(opt, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(opt)
        self.dropout                  = nn.Dropout(dropout)

    def forward(self, trg, enc_src):
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]

        #self attention
        _trg, _ = self.self_attention(trg, trg, trg)

        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        #trg = [batch size, trg len, hid dim]

        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src)

        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        #trg = [batch size, trg len, hid dim]

        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]

        return trg, attention

class SAEncoder(nn.Module):
    def __init__(self, opt, device, max_length=12):
        super().__init__()

        input_dim = opt.input_dim 
        hid_dim   = opt.hid_dim   
        n_layers  = opt.n_layers  
        n_heads   = opt.n_heads   
        pf_dim    = opt.pf_dim    
        dropout   = opt.dropout   

        self.device = device
        
        self.tok_embedding = nn.Linear(input_dim,  hid_dim)
        self.pos_embedding = nn.Linear(1, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(opt, device) 
                                     for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src):
        #src = [batch size, src len, 16]
        
        batch_size = src.shape[0]
        src_len    = src.shape[1]
        
        #pos = torch.arange(0, src_len).reshape(1, src_len, 1).repeat(batch_size, 1, 1).to(self.device)
        #pos = [batch size, src len, 1]
        
        #src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        src = self.dropout((self.tok_embedding(src) * self.scale))
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src)
        #src = [batch size, src len, hid dim]
            
        return src

class SADecoder(nn.Module):
    def __init__(self, opt, device, max_length=12):
        super().__init__()

        input_dim  = opt.input_dim
        output_dim = opt.output_dim
        hid_dim    = opt.hid_dim
        n_layers   = opt.n_layers
        n_heads    = opt.n_heads
        pf_dim     = opt.pf_dim
        dropout    = opt.dropout  
        
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

        self.tok_embedding = nn.Linear(input_dim, hid_dim)
        self.pos_embedding = nn.Linear(max_length, hid_dim)

        self.layers  = nn.ModuleList([DecoderLayer(opt, device) 
                                      for _ in range(n_layers)])
        self.fc_out  = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src):
        #trg = [batch size, trg len, 1]
        #enc_src = [batch size, src len, hid dim]

        batch_size = trg.shape[0]
        trg_len    = trg.shape[1]

        #pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #pos = [batch size, trg len]

        #trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        trg = self.dropout((self.tok_embedding(trg) * self.scale))
        #trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src)
        #trg       = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)
        #output = [batch size, trg len, output dim]

        return output, attention
    
class SelfAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        #src = [batch size, src len, 16]
        #trg = [batch size, trg len, 1]

        enc_src = self.encoder(src)
        #enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src)
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]

        return output, attention
    
