import torch
from torch.utils.data import Dataset
import os, shutil
import numpy as np
import random

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
        
        self.source_size  = config.source_size
        self.target_size  = config.target_size
        self.shuffle      = config.shuffle

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

        self.source_size  = config.source_size
        self.target_size  = config.target_size
        self.size       = self.data.shape[0] - self.source_size - self.target_size + 1

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

class PMDiscreteDataset(Dataset):
    def __init__(self, config, sitename, isTrain=False, mode='all'):

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

        self.source_size  = config.source_size
        self.target_size  = config.target_size
        self.size         = self.data.shape[0] - self.source_size - self.target_size + 1
        self.mode         = mode
        
        # Discrete ground truth on label
        # Denormalize ground truth 
        true = self.data[:, 7].copy() * self.thres_data[:, 7] // 10
        thres = self.thres_data[:, 7] // 10
        true[true>20] = 20
        thres[thres>20] = 20
        self.true = np.zeros((self.data.shape[0], 21))
        self.thres = np.zeros((self.data.shape[0], 21))
        self.ext = np.zeros((self.data.shape[0], 1))
        for i, (v, t) in enumerate(zip(true, thres)):
            v = int(v)
            t = int(t)
            self.true[i][v] = 1
            self.thres[i][t] = 1
            self.ext[i][0] = v >= t

        if mode != 'all':
            self.data_list  = []
            self.true_list  = []
            self.thres_list = []
            self.ext_list   = []
            x_len = self.source_size
            y_len = self.target_size
            for i in range(self.size):
                st = i
                ed = i + x_len
                window       = self.data[st:ed]
                st = i + x_len
                ed = i + x_len + y_len
                true_window  = self.true [st:ed]
                thres_window = self.thres[st:ed]
                ext_window   = self.ext  [st:ed]
                # if ext in y_data, put it to ext_list, or to norm_list
                if mode == 'ext'    and np.sum(ext_window) > 0:
                    self.data_list.append(window)
                    self.true_list.append(true_window)
                    self.thres_list.append(thres_window)
                    self.ext_list.append(ext_window)
                if mode == 'normal' and np.sum(ext_window) == 0:
                    self.data_list.append(window)
                    self.true_list.append(true_window)
                    self.thres_list.append(thres_window)
                    self.ext_list.append(ext_window)
            self.data_list  = np.array(self.data_list)
            self.true_list  = np.array(self.true_list)
            self.thres_list = np.array(self.thres_list)
            self.ext_list   = np.array(self.ext_list)
            self.size = self.true_list.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        """
        if self.mode == 'all':
            st = idx
            ed = idx + self.source_size
            x = self.data[st:ed]

            st = idx + self.source_size
            ed = idx + self.source_size + self.target_size
            y = self.true[st:ed]
            y_ext = self.ext[st:ed]
            thres_y = self.thres[st:ed]
        else:
            x       = self.data_list[idx]
            y       = self.true_list[idx]
            thres_y = self.thres_list[idx]
            y_ext   = self.ext_list[idx]

        return  torch.FloatTensor(x),\
                torch.FloatTensor(y),\
                torch.FloatTensor(y_ext),\
                torch.FloatTensor(thres_y)

class PMUnetDataset(Dataset):
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

        self.source_size  = config.source_size
        self.target_size  = config.target_size
        self.size         = self.data.shape[0] - self.source_size + 1
        self.isTrain      = isTrain

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        """
        st = idx
        ed = idx + self.source_size
        x = self.data[st:ed]

        y = self.data[st:ed, 7:8].copy()
        y_ext = y >= 1
        thres_y = self.thres_data[st:ed, 7:8]
        # random mask data
        if self.isTrain:
            sz = 8
            st = random.randint(0, x.shape[0]-sz)
            x[st:st+sz, :-3] = 0

        return  torch.FloatTensor(x),\
                torch.FloatTensor(y),\
                torch.FloatTensor(y_ext),\
                torch.FloatTensor(thres_y)

class PMFudanDataset(Dataset):
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

        self.data       = _read_file(mode=0) #[:, 7:8]
        self.thres_data = _read_file(mode=1) #[:, 7:8]
        # Calculate slope for adding extreme data
        dif_data = self.data[1:, 7] - self.data[:-1, 7]
        index = np.argwhere(dif_data>=15)[:, 0] + 1
        self.mask = np.zeros((self.data.shape[0], 1))
        self.mask[index] = 1

        self.model        = config.model
        self.memory_size  = config.memory_size
        self.window_size  = config.window_size
        self.source_size  = config.source_size
        self.target_size  = config.target_size
        self.input_dim    = config.input_dim
        self.isTrain      = isTrain

        self.size = self.data.shape[0] - self.memory_size - self.window_size - self.source_size - self.target_size + 1
        # Create past window input & past extreme event label
        self.all_window = np.zeros([self.size+self.memory_size, self.window_size, self.data.shape[-1]])
        self.all_ext    = np.zeros([self.size+self.memory_size, 1])
        for j in range(self.all_window.shape[0]):
            # input
            st = j
            ed = j + self.window_size
            self.all_window[j] = self.data[st: ed]
            # label
            st = j + self.window_size + self.target_size - 1
            ed = j + self.window_size + self.target_size
            self.all_ext[j] = np.logical_or(self.data[st: ed, 7:8] >= 1, self.mask[st:ed])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
            past_windows: [batch, target_size, window_len, 16]
            past_ext:     [batch, target_size, window_len, 1]
            x:            [batch, target_size, 16]
            y:            [batch, target_size, 1]
            y_ext:        [batch, target_size, 1]
        """
        # Past window, each window has a sequence of data
        past_windows = np.zeros((self.source_size, self.memory_size, self.window_size, self.input_dim))
        past_exts    = np.zeros((self.source_size, self.memory_size, 1)) 
        indexs = np.arange(idx + self.memory_size)
        for k in range(past_windows.shape[0]):
            np.random.shuffle(indexs)
            sample = indexs[:self.memory_size]
            past_windows[k] = self.all_window[sample]
            past_exts[k]    = self.all_ext[sample]
        
        # Input
        st = idx + self.memory_size + self.window_size  
        ed = idx + self.memory_size + self.window_size + self.source_size
        x = self.data[st: ed]
        # Target, only predict pm2.5, so select '7:8'
        st = idx + self.memory_size + self.window_size + self.target_size
        ed = idx + self.memory_size + self.window_size + self.target_size + self.source_size 
        y = self.data[st:ed, 7:8]
        y_ext = np.logical_or(y >= 1, self.mask[st:ed])
        thres_y = self.thres_data[st:ed, 7:8]

        return  torch.FloatTensor(x),\
                torch.FloatTensor(y),\
                torch.FloatTensor(y_ext),\
                torch.FloatTensor(thres_y),\
                torch.FloatTensor(past_windows),\
                torch.FloatTensor(past_exts)

class PMClassDataset(Dataset):
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
        # Calculate slope for adding extreme data
        dif_data = self.data[1:, 7] - self.data[:-1, 7]
        index = np.argwhere(dif_data>=15)[:, 0] + 1
        self.mask = np.zeros((self.data.shape[0], 1))
        self.mask[index] = 1

        self.source_size  = config.source_size
        self.target_size  = config.target_size
        self.memory_size  = config.memory_size
        self.size         = self.data.shape[0] - self.memory_size - self.source_size - self.target_size + 1
        self.isTrain      = isTrain

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        """
        st = idx 
        ed = idx + self.memory_size
        past_window = self.data[st:ed]
        
        st = idx + self.target_size 
        ed = idx + self.memory_size + self.target_size
        past_ext = np.logical_or(self.data[st:ed, 7:8] >= 1, self.mask[st:ed])
        
        st = idx + self.memory_size
        ed = idx + self.memory_size + self.source_size
        x = self.data[st:ed]
        
        st = idx + self.memory_size + self.source_size
        ed = idx + self.memory_size + self.source_size + self.target_size
        y = np.logical_or(self.data[st:ed, 7:8] >= 1, self.mask[st:ed])

        return  torch.FloatTensor(x),\
                torch.FloatTensor(y),\
                torch.FloatTensor(past_window),\
                torch.FloatTensor(past_ext)
