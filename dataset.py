import torch
from torch.utils.data import Dataset
import os, shutil
import numpy as np
import random
from utils import *

class PMDataset(Dataset):
    def __init__(self, opt, sitename, isTrain=False):

        self.opt = opt

        self.data       = read_file(sitename, opt, mode=0, isTrain=isTrain)
        self.thres_data = read_file(sitename, opt, mode=1, isTrain=isTrain)
        
        # Get mask label 
        self.mask = get_mask(opt, self.data, self.thres_data)
        self.data /= self.thres_data
        # Concatenate extreme event label in data
        if not opt.no_concat_label:
            self.data = np.concatenate((self.data, self.mask), axis=-1)
        # Get split data
        if opt.split_dataset:
            assert opt.split_mode != None, f"split dataset method should setting split_mode e.x: --split_mode 'norm'"
            self.data = get_split_dataset(opt, self.data)
            self.size = self.data.shape[0]
        else:
            self.size = self.data.shape[0] - opt.memory_size - opt.source_size - opt.target_size + 1

    def get_ratio(self):
        return 1-np.sum(self.mask) / self.mask.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        """
        if self.opt.split_dataset:
            past_window = self.data[idx][: self.opt.memory_size]
            x           = self.data[idx][self.opt.memory_size: self.opt.memory_size+self.opt.source_size]
            y           = self.data[idx][self.opt.memory_size+self.opt.target_size: self.opt.memory_size+self.opt.source_size+self.opt.target_size, 7: 8]
            y_ext       = self.data[idx][self.opt.memory_size+self.opt.target_size: self.opt.memory_size+self.opt.source_size+self.opt.target_size, -1: ]
        else:
            st = idx 
            ed = idx + self.opt.memory_size
            past_window = self.data[st: ed]
            
            st = idx + self.opt.memory_size
            ed = idx + self.opt.memory_size + self.opt.source_size
            x = self.data[st: ed]

            st = idx + self.opt.memory_size + self.opt.target_size
            ed = idx + self.opt.memory_size + self.opt.source_size + self.opt.target_size
            y = self.data[st: ed, 7: 8]
            y_ext = self.data[st: ed, -1: ]

        return  torch.FloatTensor(x),\
                torch.FloatTensor(y),\
                torch.FloatTensor(y_ext),\
                torch.FloatTensor(past_window)

class PMFudanDataset(Dataset):
    def __init__(self, opt, sitename, isTrain=False):
        
        self.opt     = opt

        self.data       = read_file(sitename, opt, mode=0, isTrain=isTrain)
        self.thres_data = read_file(sitename, opt, mode=1, isTrain=isTrain)

        # Get mask label 
        self.mask = get_mask(opt, self.data, self.thres_data)
        self.data /= self.thres_data
        # Concatenate extreme event label in data
        if not opt.no_concat_label:
            self.data = np.concatenate((self.data, self.mask), axis=-1)

        self.size = self.data.shape[0] - self.opt.memory_size - self.opt.window_size - self.opt.source_size - self.opt.target_size + 1
        # Create past window input & past extreme event label
        self.all_window = np.zeros([self.size + self.opt.memory_size, self.opt.window_size, self.data.shape[-1]])
        self.all_ext    = np.zeros([self.size + self.opt.memory_size, 1])
        for j in range(self.all_window.shape[0]):
            # input
            st = j
            ed = j + self.opt.window_size
            self.all_window[j] = self.data[st: ed]
            # label
            st = j + self.opt.window_size + self.opt.target_size - 1
            ed = j + self.opt.window_size + self.opt.target_size
            self.all_ext[j] = self.mask[st:ed]

    def get_ratio(self):
        return 1-np.sum(self.mask) / self.mask.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
            past_windows: [batch, memory_size, window_len, 16]
            past_ext:     [batch, memory_size, window_len, 1]
            x:            [batch, source_size, 16]
            y:            [batch, source_size, 1]
            y_ext:        [batch, source_size, 1]
        """
        # Past window, each window has a sequence of data
        indexs = np.arange(idx + self.opt.memory_size)
        np.random.shuffle(indexs)
        sample = indexs[:self.opt.memory_size]
        past_window = self.all_window[sample]
        past_ext    = self.all_ext   [sample]
        # Input
        st = idx + self.opt.memory_size + self.opt.window_size  
        ed = idx + self.opt.memory_size + self.opt.window_size + self.opt.source_size
        x = self.data[st: ed]
        # Target, only predict pm2.5, so select '7:8'
        st = idx + self.opt.memory_size + self.opt.window_size + self.opt.target_size
        ed = idx + self.opt.memory_size + self.opt.window_size + self.opt.target_size + self.opt.source_size 
        y       = self.data      [st: ed, 7:8]
        y_ext   = self.mask      [st: ed]
        thres_y = self.thres_data[st: ed, 7:8]

        return  torch.FloatTensor(x),\
                torch.FloatTensor(y),\
                torch.FloatTensor(y_ext),\
                torch.FloatTensor(thres_y),\
                torch.FloatTensor(past_window),\
                torch.FloatTensor(past_ext)
