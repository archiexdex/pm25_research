import torch
from torch.utils.data import Dataset
from torch import Tensor, FloatTensor
import numpy as np
import random
import json
import os, shutil
import warnings

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
        self.target_size = config.target_size
        self.threshold = config.threshold
        self.size = len(self.data) - self.memory_size - self.window_size - self.target_size 
        self.mean = {}
        self.std = {}
        with open(config.mean_path, "r") as fp:
            self.mean = json.load(fp)[sitename]
        with open(config.std_path, "r") as fp:
            self.std = json.load(fp)[sitename]
        #print(self.data[self.memory_size+self.window_size+self.target_size, -3:])
        #input("!@#")
        # Normalize data
        self.data_copy = self.data.copy()
        self.data = (self.data - self.mean) / self.std
        
        self.threshold = ( self.threshold - self.mean[7]) / self.std[7]
        # Create past window input & past extreme event label
        self.all_window = np.zeros([self.size+self.memory_size, self.window_size, 16])
        self.all_ext    = np.zeros([self.size+self.memory_size, 1])
        for j in range(self.all_window.shape[0]):
            self.all_window[j] = self.data[j: j+self.window_size]
            self.all_ext[j]    = self.data[j+self.window_size: j+self.window_size+1, 7:8] > self.threshold

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
            past_window: [batch, window_size, window_len, 16]
            past_ext: [batch, window_size, 1]
            x: [batch, target_len, 16]
            y: [batch, target_len, 1]
            y_ext: [batch, target_len, 1]
        """
        # Past window, each window has a sequence of data
        st = idx
        ed = idx + self.memory_size 
        past_window = self.all_window[st: ed]
        past_ext = self.all_ext[st: ed]
        
        # Input
        st = idx + self.memory_size + self.window_size
        ed = idx + self.memory_size + self.window_size + self.target_size
        x = self.data[st: ed]
        # Target, only predict pm2.5, so select '7:8'
        st = idx + self.memory_size + self.window_size + self.target_size
        ed = idx + self.memory_size + self.window_size + self.target_size + 1
        y = self.data[st: ed, 7:8]
        y_ext = y > self.threshold

        return  torch.FloatTensor(x),\
                torch.FloatTensor(y),\
                torch.FloatTensor(y_ext),\
                torch.FloatTensor(past_window),\
                torch.FloatTensor(past_ext)
        
