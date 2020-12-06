import torch
from torch.utils.data import Dataset
from torch import Tensor, FloatTensor
import numpy as np
import random
import json
import os, shutil

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

class PMSingleSiteDataset(Dataset):
    def __init__(self, sitename='美濃', target_hour=8, isTrain=False):
        
        filename = f"dataset/norm/train/{sitename}.npy" if isTrain else f"dataset/norm/valid/{sitename}.npy"
        if os.path.exists(filename):
            self.data = np.load(filename) 
        else:
            raise ValueError(f"path {filename} doesn't exist")

        self.target_hour = target_hour 
        self.sz = len(self.data) - target_hour 

    def __len__(self):
        return self.sz 

    def __getitem__(self, idx):
        # input, target 
        x = FloatTensor(self.data[idx:idx+self.target_hour])
        y = FloatTensor(self.data[idx+self.target_hour, 7:8])
        return x, y 
        
#    def collate_fn(insts):
#        pass
