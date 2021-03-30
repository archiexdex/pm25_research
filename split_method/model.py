import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, source_size):
        super().__init__()
        self.dense_emb = nn.Linear(input_dim, emb_dim)
        self.dense_hid = nn.Linear(emb_dim, hid_dim)
        self.dense_out = nn.Linear(hid_dim*source_size, output_dim)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(True)
        self.leakyrelu = nn.LeakyReLU(True)
    
    def forward(self, x):
        source_size = x.shape[1]
        x = self.leakyrelu(self.dense_emb(x))
        x = self.leakyrelu(self.dense_hid(self.dropout(x)))
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = self.relu(self.dense_out(self.dropout(x)))
        return x

class GRU_merged(nn.Module):
    def __init__(self, ext_model, nor_model, output_dim):
        super().__init__()
        self.ext_model = ext_model
        self.nor_model = nor_model
        self.dense_out = nn.Linear(output_dim * 2, output_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        source_size = x.shape[1]
        ext = self.ext_model(x)
        nor = self.nor_model(x)
        total = torch.cat((ext, nor), dim=-1)
        out = self.dense_out(total)
        return out

class DNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, source_size):
        super().__init__()
        self.dense_emb = nn.Linear(input_dim, emb_dim)
        self.dense_hid = nn.Linear(emb_dim, hid_dim)
        self.dense_out = nn.Linear(hid_dim*source_size, output_dim)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(True)
        self.leakyrelu = nn.LeakyReLU(True)
    
    def forward(self, x):
        source_size = x.shape[1]
        x = self.leakyrelu(self.dense_emb(x))
        x = self.leakyrelu(self.dense_hid(self.dropout(x)))
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = self.relu(self.dense_out(self.dropout(x)))
        return x

class DNN_merged(nn.Module):
    def __init__(self, ext_model, nor_model, output_dim):
        super().__init__()
        self.ext_model = ext_model
        self.nor_model = nor_model
        self.dense_out = nn.Linear(output_dim * 2, output_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        source_size = x.shape[1]
        ext = self.ext_model(x)
        nor = self.nor_model(x)
        total = torch.cat((ext, nor), dim=-1)
        out = self.dense_out(total)
        return out
