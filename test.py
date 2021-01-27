from utils import *
from tqdm import tqdm
import os, shutil 
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import DataLoader 
import argparse
from constants import * 

from model import * 
from dataset.dataset import PMSingleSiteDataset 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
opt = parse()

if opt.no is not None:
    no = opt.no 
else: 
    print("n is not a number")
    exit() 

cpt_dir = opt.cpt_dir 
save_dir = opt.test_results_dir 
cpt_dir = os.path.join(cpt_dir, f"{no}")

if not os.path.exists(cpt_dir):
    print("Are you kidding me??? (,,ﾟДﾟ)")
    exit() 
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_dir = os.path.join(save_dir, f"{no}")
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.mkdir(save_dir) 

mean_dict = {}
std_dict = {}
with open("dataset/train_mean.json", "r") as fp:
    mean_dict = json.load(fp)
with open("dataset/train_std.json", "r") as fp:
    std_dict = json.load(fp)

for name in sitenames:
    print(f"sitename: {name}")
    sitename = name 
    valid_dataset = PMSingleSiteDataset(sitename=sitename, target_hour=8, target_length=8, isTrain=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    mean = mean_dict[sitename][7]
    std = std_dict[sitename][7]

    # Load model
    #model = SimpleRNN(target_length=8)
    model = Seq2Seq(
                input_dim=15,
                emb_dim=64,
                output_dim=1,
                hid_dim=64,
                device=device,
                dropout=0.6,
                bidirectional=True,
            )
    checkpoint = torch.load(os.path.join(cpt_dir, f"{sitename}.pt"))
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
        o = model.interface(x)
        loss = criterion(o, y)
        sum_loss += loss.item()
        # append result
        # Denorm the data
        o = o[0,0,0]
        real_o = o.item() * std + mean 
        predict_result.append(real_o)
        trange.set_description(f"testing mean loss: {sum_loss / (idx+1):.4f}")

    valid_loss = sum_loss / len(valid_dataloader) 
        
    # Save results
    np.save(f"{save_dir}/{sitename}.npy", predict_result)
