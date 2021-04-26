import torch
from torch.utils.data import Dataset
import os, shutil
import numpy as np

class PMExtDataset(Dataset):
    def __init__(self, config, sitename, use_ext, isTrain=False):
        
        def _read_file(mode):
            if mode == 0:
                read_path = os.path.join(config.norm_train_dir, f"{sitename}.npy") if isTrain else os.path.join(config.norm_valid_dir, f"{sitename}.npy")
            elif mode == 1:
                read_path = os.path.join(config.thres_train_dir, f"{sitename}.npy") if isTrain else os.path.join(config.thres_valid_dir, f"{sitename}.npy")
            if os.path.exists(read_path):
                data = np.load(read_path)
            else:
                raise ValueError(f"path {filename} doesn't exist")
            return data
        
        data       = _read_file(mode=0)
        thres_data = _read_file(mode=1)
        
        self.model        = config.model
        self.memory_size  = config.memory_size
        self.window_size  = config.window_size
        self.source_size  = config.source_size
        self.target_size  = config.target_size
        self.threshold    = config.threshold
        self.shuffle      = config.shuffle
        self.is_transform = config.is_transform

        data_list  = []
        thres_list = []
        x_len = self.source_size
        y_len = self.target_size
        for i in range(data.shape[0]-(x_len+y_len)):
            st = i
            ed = i + x_len + y_len
            window = data[st:ed]
            thres_window = thres_data[st:ed]
            y_data = window[x_len:, 7]
            # if ext in y_data, put it to ext_list, or to norm_list
            if use_ext and np.sum(y_data[y_data>=1]) > 0:
                data_list.append(window)
                thres_list.append(thres_window)
            if not use_ext and np.sum(y_data[y_data<1]) > 0:
                data_list.append(window)
                thres_list.append(thres_window)
        self.data       = np.array(data_list)
        self.thres_data = np.array(thres_list)
        self.size       = self.data.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        """
        window = self.data[idx]
        x = window[:self.source_size]
        y = window[self.source_size:, 7:8]
        y_ext = y >= 1
        thres_window = self.thres_data[idx]
        thres_y = thres_window[self.source_size:, 7:8]

        return  torch.FloatTensor(x),\
                torch.FloatTensor(y),\
                torch.FloatTensor(y_ext),\
                torch.FloatTensor(thres_y)

class PMDataset(Dataset):
    def __init__(self, config, sitename, isTrain=False):

        def _read_file(mode):
            if mode == 0:
                read_path = os.path.join(config.norm_train_dir, f"{sitename}.npy") if isTrain else os.path.join(config.norm_valid_dir, f"{sitename}.npy")
            elif mode == 1:
                read_path = os.path.join(config.thres_train_dir, f"{sitename}.npy") if isTrain else os.path.join(config.thres_valid_dir, f"{sitename}.npy")
            if os.path.exists(read_path):
                data = np.load(read_path)
            else:
                raise ValueError(f"path {filename} doesn't exist")
            return data

        self.data       = _read_file(mode=0)
        self.thres_data = _read_file(mode=1)

        self.model        = config.model
        self.memory_size  = config.memory_size
        self.window_size  = config.window_size
        self.source_size  = config.source_size
        self.target_size  = config.target_size
        self.threshold    = config.threshold
        self.shuffle      = config.shuffle
        self.is_transform = config.is_transform
        self.size       = self.data.shape[0] - self.source_size - self.target_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        """
        st = idx
        ed = idx + self.source_size
        x = self.data[st:ed]

        st = idx + self.source_size
        ed = idx + self.source_size + self.target_size
        y = self.data[st:ed, 7:8]
        y_ext = y >= 1
        thres_y = self.thres_data[st:ed, 7:8]

        return  torch.FloatTensor(x),\
                torch.FloatTensor(y),\
                torch.FloatTensor(y_ext),\
                torch.FloatTensor(thres_y)
