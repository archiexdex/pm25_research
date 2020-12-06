from tqdm import tqdm
import os, shutil 
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import DataLoader 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model import Model 
from dataset.dataset import PMSingleSiteDataset 

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

feature_cols = ['SO2', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5',
                'RAINFALL', 'RH', 'AMB_TEMP', 'WIND_cos', 'WIND_sin',
                'hour', 'month' 
                ]
# Load data
cpt_dir = 'model_weights/'
save_dir = 'predict_results/'
if os.path.exists(save_dir):
    os.mkdir(save_dir)

mean_dict = {}
std_dict = {}
with open("dataset/train_mean.json", "r") as fp:
    mean_dict = json.load(fp)
with open("dataset/train_std.json", "r") as fp:
    std_dict = json.load(fp)

for name in sitenames:
    sitename = name 
    valid_dataset = PMSingleSiteDataset(sitename=sitename, target_hour=8, isTrain=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    mean = mean_dict[sitename][7]
    std = std_dict[sitename][7]

    # Load model 
    model = Model()
    checkpoint = torch.load(os.path.join(save_dir, f"{sitename}.pt"))
    model.load_state_dict(checkpoint) 
    model.to(device)
    criterion = nn.MSELoss()

    predict_result = []
    sum_loss = 0
    model.eval()
    trange = tqdm(valid_dataloader)
    for idx, data in enumerate(trange):
        # get data
        x, y = data 
        x = x.to(device) 
        y = y.to(device) 
        # get loss & update
        o = model(x)
        loss = criterion(o, y)
        sum_loss += loss.item()
        # append result
        # Denorm the data
        real_o = o.item() * std + mean 
        predict_result.append(real_o)
        trange.set_description(f"testing mean loss: {sum_loss / (idx+1):.4f}")

    valid_loss = sum_loss / len(valid_dataloader) 
        
    # Save results
    np.save(f"{save_dir}/{sitename}.npy", predict_result)
