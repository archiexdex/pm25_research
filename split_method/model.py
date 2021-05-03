import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_loss import *

class GRU(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, source_size, dropout, num_layers, bidirectional):
        super().__init__()
        self.dense_emb = nn.Linear(input_dim, emb_dim)
        self.dense_out = nn.Linear(hid_dim*source_size*num_layers, output_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
    
    def forward(self, x):
        source_size = x.shape[1]
        x = self.relu(self.dense_emb(x))
        #x = self.dropout(x)
        x, _ = self.rnn(x)
        x = self.relu(x)
        x = x.reshape(-1, x.shape[1] * x.shape[2])
        x = self.relu(self.dense_out(self.dropout(x)))
        return x

class DNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, source_size):
        super().__init__()
        self.dense_emb = nn.Linear(input_dim, emb_dim)
        self.dense_hid = nn.Linear(emb_dim, hid_dim)
        self.dense_out = nn.Linear(hid_dim*source_size, output_dim)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.activation = EXTActvation()
        self.softmax = nn.Softmax(dim=-1)
        self.leakyrelu = nn.LeakyReLU()
    
    def forward(self, x):
        source_size = x.shape[1]
        x = self.relu(self.dense_emb(x))
        x = self.relu(self.dense_hid(x))
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = self.relu(self.dense_out(self.dropout(x)))
        return x

class DNN_merged(nn.Module):
    def __init__(self, ext_model, nor_model, input_dim, output_dim, source_size):
        super().__init__()
        self.ext_model = ext_model
        self.nor_model = nor_model
        self.dense_emb = nn.Linear(input_dim, output_dim)
        self.dense_out = nn.Linear(source_size+2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        source_size = x.shape[1]
        emb = self.dense_emb(x).squeeze(-1)
        ext = self.ext_model(x)
        nor = self.nor_model(x)
        total = torch.cat((emb, ext , nor ), dim=-1)
        #out = emb + ext * 0.95 + nor * 0.05
        out = self.dense_out(total)
        return out

class G_DNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, source_size):
        super().__init__()
        self.dense_emb = nn.Linear(input_dim, emb_dim)
        self.dense_hid = nn.Linear(emb_dim, hid_dim)
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
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, source_size, dropout, num_layers, bidirectional):
        super().__init__()
        self.dense_emb = nn.Linear(input_dim, emb_dim)
        self.dense_out = nn.Linear(hid_dim*num_layers, output_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
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
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, source_size):
        super().__init__()
        self.dense_emb = nn.Linear(input_dim, emb_dim)
        self.dense_hid = nn.Linear(emb_dim, hid_dim)
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
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, source_size, dropout, num_layers, bidirectional):
        super().__init__()
        self.dense_emb = nn.Linear(input_dim, emb_dim)
        self.dense_out = nn.Linear(hid_dim*source_size*num_layers, output_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
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
