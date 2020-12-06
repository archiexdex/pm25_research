from tqdm import tqdm
import os, shutil 
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader 
from datetime import datetime 

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

no = 0 
save_dir = 'model_weights/'
log_dir = 'logs'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
if os.path.exists(f"{log_dir}/{no}"):
    os.remove(f"{log_dir}/{no}")

def update_model(loss_function, optimizer, output, target, retain_graph=False):
    loss = loss_function(output, target)
    optimizer.zero_grad()
    loss.backward(retain_graph=retain_graph)
    optimizer.step()
    return loss

############ train model #############
for name in sitenames:
    #sitename = '美濃'
    sitename = name 
    print(sitename)
    train_dataset = PMSingleSiteDataset(sitename=sitename, target_hour=8, isTrain=True)
    valid_dataset = PMSingleSiteDataset(sitename=sitename, target_hour=8, isTrain=False)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    model = Model()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    max_epochs = 100
    best_loss = 1e9
    patience = 5
    earlystop_counter = 0

    for epoch in range(max_epochs):
        print(f">>Epoch: {epoch}\n")
        model.train()
        sum_loss = 0
        trange = tqdm(train_dataloader)
        for idx, data in enumerate(trange):
            # get data
            x, y = data 
            x = x.to(device) 
            y = y.to(device) 
            # get loss & update
            o = model(x)
            loss = update_model(criterion, optimizer, o, y)
            sum_loss += loss.item()
            trange.set_description(f"Training mean loss: {sum_loss / (idx+1):.4f}")
        train_loss = sum_loss / len(train_dataloader)

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
            trange.set_description(f"validing mean loss: {sum_loss / (idx+1):.4f}")
        
        valid_loss = sum_loss / len(valid_dataloader) 
        
        if best_loss > valid_loss:
            best_loss = valid_loss 
            torch.save(model.state_dict(), f"checkpoint.pt")
            earlystop_counter = 0
            print(">> Model saved!!")
                
        # if best_loss doesn't improve for patience times, terminate training
        else:
            earlystop_counter += 1
            if earlystop_counter >= patience:
                print("Early stop!!!")
                print(f"sitename: {sitename}\nepoch: {epoch}\nbest_loss: {best_loss:.3f}")
                os.rename("checkpoint.pt", os.path.join(save_dir, f"{sitename}.pt"))
                # write log
                with open(f"{log_dir}/{no}", "a") as fp:
                    fp.write(f">>sitename: {sitename}\nepoch: {epoch}\nbest_loss: {best_loss:.3f}\ntimestamp: {datetime.now()}\n\n")
                break
