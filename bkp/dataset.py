from constants import * 
import torch
from torch.utils.data import Dataset
from torch import Tensor, FloatTensor
import numpy as np
import random
import json
import os, shutil
#from lmoments3 import distr

class PMMultiSiteDataset(Dataset):
    def __init__(self, config, sitenames, isTrain=False):
        
        self.model        = config.model
        self.memory_size  = config.memory_size
        self.window_size  = config.window_size
        self.source_size  = config.source_size
        self.target_size  = config.target_size
        self.shuffle      = config.shuffle
        self.is_transform = config.is_transform

        with open(config.mean_path, "r") as fp:
            mean_dict = json.load(fp)
        with open(config.std_path, "r") as fp:
            std_dict = json.load(fp)
        with open(config.threshold_path, "r") as fp:
            threshold_dict = json.load(fp)

        self.data, self.data_copy, self.y_true, self.thres_list = [[] for _ in range(4)]
        self.data_copy = []
        for sitename in sitenames:
            filename = f"data/norm/train/{sitename}.npy" if isTrain else f"data/norm/valid/{sitename}.npy"
            if os.path.exists(filename):
                 data = np.load(filename) 
            else:
                raise ValueError(f"path {filename} doesn't exist")
            
            data_copy = data.copy()
            # summer threshold
            s_index = np.isin(data[:, -3], summer_months)
            # winter threshold
            w_index = np.isin(data[:, -3], summer_months, invert=True)
            # create y_true
            y_true = data[:, 7:8].copy()
            thres_list = np.zeros((data.shape[0], 1))
            s_threshold = threshold_dict[sitename]["summer"]
            w_threshold = threshold_dict[sitename]["winter"]
            thres_list[s_index] = s_threshold
            thres_list[w_index] = w_threshold
            # Append each data to list
            self.data.append(data)
            self.data_copy.append(data_copy)
            self.y_true.append(y_true)
            self.thres_list.append(thres_list)
        self.size       = len(self.data[0]) - self.memory_size - self.source_size - self.target_size
        self.data       = np.array(self.data)
        self.data_copy  = np.array(self.data_copy)
        self.y_true     = np.array(self.y_true)
        self.thres_list = np.array(self.thres_list)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
            past_window: [batch, site_size, window_len, 16]
            past_ext:    [batch, site_size, window_len, 1]
            x:           [batch, site_size, target_len, 16]
            y:           [batch, site_size, target_len, 8]
            y_ext:       [batch, site_size, target_len, 8]
        """
        # Past window, each window has a sequence of data
        st = idx
        ed = idx + self.memory_size
        past_window = self.data[:, st: ed]
        past_ext    = self.y_true[:, st: ed]
        
        # Input
        st = idx + self.memory_size 
        ed = idx + self.memory_size + self.source_size
        x = self.data[:, st: ed]
        # Target, only predict pm2.5, so select '7:8'
        st = idx + self.memory_size + self.source_size 
        ed = idx + self.memory_size + self.source_size + self.target_size
        y       = self.data[:, st: ed, 7:8]
        y_ext   = self.y_true[:, st: ed]
        y_thres = self.thres_list[:, st: ed]

        # For CNN
        if self.is_transform > 0:
            past_window = np.transpose(past_window, (2, 0, 1))
            past_ext = np.transpose(past_ext, (2, 0, 1))
            x = np.transpose(x, (2, 0, 1))
            y = np.transpose(y, (2, 0, 1))
            y_ext = np.transpose(y_ext, (2, 0, 1))
            y_thres = np.transpose(y_thres, (2, 0, 1))


        return  torch.FloatTensor(x),\
                torch.FloatTensor(y),\
                torch.FloatTensor(y_ext),\
                torch.FloatTensor(past_window),\
                torch.FloatTensor(past_ext), \
                torch.FloatTensor(y_thres) # For testing to check the values
    
class PMExtDataset(Dataset):
    def __init__(self, config, sitename, isTrain=False):
        
        read_path = os.path.join(config.norm_train_dir, f"{sitename}.npy") if is_Train else os.path.join(config.norm_valid_dir, f"{sitename}.npy")
        if os.path.exists(read_path):
            data = np.load(read_path) 
        else:
            raise ValueError(f"path {filename} doesn't exist")

        self.model        = config.model
        self.memory_size  = config.memory_size
        self.window_size  = config.window_size
        self.source_size  = config.source_size
        self.target_size  = config.target_size
        self.threshold    = config.threshold
        self.shuffle      = config.shuffle
        self.is_transform = config.is_transform
        self.use_ext_data = config.use_ext_data

        data_list = []
        for i in range(0, data.shape[0]-(x_len+y_len)):
            st = i
            ed = i + x_len + y_len
            window = data[st: ed]
            y_data = window[x_len:, 7]
            # if ext in y_data, put it to ext_list, or to norm_list
            if self.use_ext_data == 1 and np.sum(y_data[y_data>=1]) > 0:
                data_list.append(window)
            else:
                data_list.append(window)
        self.data  = np.array(data_list)
        self.size = self.data.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        """
        window = self.data[idx]
        x = window[:self.source_size]
        y = window[self.source_size:, 7]
        y_ext = y >= 1

        return  torch.FloatTensor(x),\
                torch.FloatTensor(y),\
                torch.FloatTensor(y_ext),\
    
class PMSingleSiteDataset(Dataset):
    def __init__(self, config, sitename, isTrain=False):
        
        filename = f"data/origin/train/{sitename}.npy" if isTrain else f"data/origin/valid/{sitename}.npy"
        if os.path.exists(filename):
            self.origin_data = np.load(filename) 
        else:
            raise ValueError(f"path {filename} doesn't exist")
        
        filename = f"data/norm/train/{sitename}.npy" if isTrain else f"data/norm/valid/{sitename}.npy"
        if os.path.exists(filename):
            self.data = np.load(filename) 
        else:
            raise ValueError(f"path {filename} doesn't exist")
        
        self.model        = config.model
        self.memory_size  = config.memory_size
        self.window_size  = config.window_size
        self.source_size  = config.source_size
        self.target_size  = config.target_size
        self.threshold    = config.threshold
        self.shuffle      = config.shuffle
        self.is_transform = config.is_transform

        if self.model == "fudan":
            self.size = len(self.data) - self.memory_size - self.window_size - self.source_size - self.target_size
        else:
            self.size = len(self.data) - self.memory_size - self.source_size - self.target_size
        self.mean = {}
        self.std = {}
        self.threshold = {}
        with open(config.mean_path, "r") as fp:
            self.mean = json.load(fp)[sitename]
        with open(config.std_path, "r") as fp:
            self.std = json.load(fp)[sitename]
        with open(config.threshold_path, "r") as fp:
            self.threshold = json.load(fp)[sitename]
        # Backup the data
        #self.data_copy = self.data.copy()
        # summer threshold
        s_index = np.isin(self.origin_data[:, -3], summer_months)
        # winter threshold
        w_index = np.isin(self.origin_data[:, -3], summer_months, invert=True)
        # create y_true
        y_true = self.data[:, 7:8].copy()
        thres_list = np.zeros((self.data.shape[0], 1))
        thres_list[s_index] = self.threshold["summer"]
        thres_list[w_index] = self.threshold["winter"]
        self.y_true = y_true
        self.thres_list = thres_list
        # Create past window input & past extreme event label
        self.all_window = np.zeros([self.size+self.memory_size, self.window_size, len(feature_cols)])
        self.all_ext    = np.zeros([self.size+self.memory_size, 1])
        for j in range(self.all_window.shape[0]):
            self.all_window[j] = self.data[j: j+self.window_size]
            st = j + self.window_size + self.target_size - 1
            ed = j + self.window_size + self.target_size
            self.all_ext[j] = self.y_true[st: ed] >= 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
            past_window: [batch, window_len, 16]
            past_ext: [batch, window_len, 1]
            x: [batch, target_len, 16]
            y: [batch, target_len, 81]
            y_ext: [batch, target_len, 8]
        """
        # Past window, each window has a sequence of data
        if self.model == "fudan":
            st = idx
            ed = idx + self.memory_size 
            past_window = self.all_window[st: ed]
            past_ext = self.all_ext[st: ed]
            # shuffle window
            indexs = np.arange(self.memory_size)
            if self.shuffle > 0:
                np.random.shuffle(indexs)
                past_window = past_window[indexs]
                past_ext = past_ext[indexs]
        else:
            st = idx
            ed = idx + self.memory_size
            past_window = self.data[st: ed]
            past_ext = self.y_true[st: ed]
        
        # Input
        st = idx + self.memory_size 
        ed = idx + self.memory_size + self.source_size
        x = self.data[st: ed]
        # Target, only predict pm2.5, so select '7:8'
        st = idx + self.memory_size + self.source_size 
        ed = idx + self.memory_size + self.source_size + self.target_size
        y = self.data[st: ed, 7:8]
        y_ext = self.y_true[st: ed] >= 1

        # info
        st = idx + self.memory_size
        ed = idx + self.memory_size + self.source_size + self.target_size
        xy_thres = self.thres_list[st: ed]

        # For CNN
        if self.is_transform > 0:
            past_window = np.transpose(past_window, (1, 0))
            past_ext = np.transpose(past_ext, (1, 0))
            x = np.transpose(x, (1, 0))
            y = np.transpose(y, (1, 0))
            y_ext = np.transpose(y_ext, (1, 0))


        return  torch.FloatTensor(x),\
                torch.FloatTensor(y),\
                torch.FloatTensor(y_ext),\
                torch.FloatTensor(past_window),\
                torch.FloatTensor(past_ext),\
                torch.FloatTensor(xy_thres) # For testing to check the values

    
#    def get_gev_params(self):
#        x = self.data[:, 7]
#        params = distr.gev.lmom_fit(x)
#        mean = params['loc']
#        std = params['scale']
#        shape = params['c']
#        return mean, std, shape
