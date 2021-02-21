import torch
from torch.utils.data import Dataset
from torch import Tensor, FloatTensor
import numpy as np
import random
import json
import os, shutil
from lmoments3 import distr

# size of 73
sitenames = [
    '三義', '三重', '中壢', '中山', '二林', '仁武', '冬山', '前金', '前鎮', '南投', 
    '古亭', '善化', '嘉義', '土城', '埔里', '基隆', '士林', '大同', '大園', '大寮', 
    '大里', '安南', '宜蘭', '小港', '屏東', '崙背', '左營', '平鎮', '彰化', '復興', 
    '忠明', '恆春', '斗六', '新店', '新港', '新營', '新竹', '新莊', '朴子', '松山', 
    '板橋', '林口', '林園', '桃園', '楠梓', '橋頭', '永和', '汐止', '沙鹿', '淡水', 
    '湖口', '潮州', '竹山', '竹東', '線西', '美濃', '臺南', '臺東', '臺西', '花蓮', 
    '苗栗', '菜寮', '萬華', '萬里', '西屯', '觀音', '豐原', '關山', '陽明', '頭份', 
    '鳳山', '麥寮', '龍潭'
    ]
sitenames_sorted = sorted(sitenames)

feature_cols = ['SO2', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5',
                'RAINFALL', 'RH', 'AMB_TEMP', 'WIND_cos', 'WIND_sin',
                'month', 'day', 'hour'
                ]
class PMSingleSiteDataset(Dataset):
    def __init__(self, config, sitename, isTrain=False):
        
        filename = f"dataset/origin/train/{sitename}.npy" if isTrain else f"dataset/origin/valid/{sitename}.npy"
        if os.path.exists(filename):
            self.data = np.load(filename) 
        else:
            raise ValueError(f"path {filename} doesn't exist")

        self.memory_size = config.memory_size
        self.window_size = config.window_size
        self.source_size = config.source_size
        self.target_size = config.target_size
        self.threshold = config.threshold
        self.shuffle = config.shuffle
        self.size = len(self.data) - self.memory_size - self.source_size - self.target_size + 1
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
        self.data_copy = self.data.copy()
        # summer threshold
        s_index = np.isin(self.data[:, -3], [4,5,6,7,8,9])
        # winter threshold
        w_index = np.isin(self.data[:, -3], [4,5,6,7,8,9], invert=True)
        # create y_true
        y_true = self.data[:, 7:8].copy()
        thres_list = np.zeros((self.data.shape[0], 1))
        s_threshold = self.threshold["summer"]
        w_threshold = self.threshold["winter"]
        s_data = y_true[s_index]
        w_data = y_true[w_index]
        s_data[s_data <= s_threshold] = 0
        s_data[s_data > s_threshold] = 1
        w_data[w_data <= w_threshold] = 0
        w_data[w_data > w_threshold] = 1
        y_true[s_index] = s_data
        y_true[w_index] = w_data
        thres_list[s_index] = s_threshold
        thres_list[w_index] = w_threshold
        self.y_true = y_true
        self.thres_list = thres_list
        # Normalize data
        self.data = (self.data - self.mean) / self.std
        # Create past window input & past extreme event label
        #self.all_window = np.zeros([self.size+self.memory_size, self.window_size, 16])
        #self.all_ext    = np.zeros([self.size+self.memory_size, 1])
        #for j in range(self.all_window.shape[0]):
        #    self.all_window[j] = self.data[j: j+self.window_size]
        #    st = j + self.window_size + self.target_size - 1
        #    ed = j + self.window_size + self.target_size
        #    if st in self.s_index:
        #        threshold = self.s_threshold
        #    else:
        #        threshold = self.w_threshold
        #    self.all_ext[j] = self.data[st: ed, 7:8] > threshold

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
            past_window: [batch, window_len, 16]
            past_ext: [batch, window_len, 1]
            x: [batch, target_len, 16]
            y: [batch, target_len, 1]
            y_ext: [batch, target_len, 1]
        """
        # Past window, each window has a sequence of data
        #st = idx
        #ed = idx + self.memory_size 
        #past_window = self.all_window[st: ed]
        #past_ext = self.all_ext[st: ed]
        ## shuffle window
        #indexs = np.arange(self.memory_size)
        #if self.shuffle:
        #    np.random.shuffle(indexs)
        #    past_window = past_window[indexs]
        #    past_ext = past_ext[indexs]
        st = idx
        ed = idx + self.memory_size
        past_window = self.data[st: ed]
        past_ext = self.y_true[st: ed]
        
        # Input
        st = idx + self.memory_size 
        ed = idx + self.memory_size + self.source_size
        x = self.data[st: ed]
        # Target, only predict pm2.5, so select '7:8'
        st = idx + self.memory_size + self.source_size + self.target_size - 1
        ed = idx + self.memory_size + self.source_size + self.target_size
        y = self.data[st: ed, 7:8]
        y_ext = self.y_true[st: ed]
        y_thres = self.thres_list[st: ed]

        return  torch.FloatTensor(x),\
                torch.FloatTensor(y),\
                torch.FloatTensor(y_ext),\
                torch.FloatTensor(past_window),\
                torch.FloatTensor(past_ext), \
                torch.FloatTensor(y_thres)
    
    def get_gev_params(self):
        x = self.data[:, 7]
        params = distr.gev.lmom_fit(x)
        mean = params['loc']
        std = params['scale']
        shape = params['c']
        return mean, std, shape
