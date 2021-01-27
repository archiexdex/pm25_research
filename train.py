from utils import *
from tqdm import tqdm
import os, shutil 
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader 
from datetime import datetime 
import argparse
from model import * 
from dataset.dataset import PMSingleSiteDataset 
from constants import *
import csv 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
opt = parse()
same_seeds(opt.seed)

if opt.no is not None:
    no = opt.no 
else: 
    print("n is not a number")
    exit() 

cpt_dir = opt.cpt_dir 
log_dir = opt.log_dir 
if not os.path.exists(cpt_dir):
    os.mkdir(cpt_dir)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

cpt_dir = os.path.join(cpt_dir, f"{no}")
if os.path.exists(cpt_dir):
    shutil.rmtree(cpt_dir)
os.mkdir(cpt_dir) 

log_file = os.path.join(log_dir, f"{no}.csv")
if os.path.exists(log_file):
    os.remove(log_file)
    with open(log_file, "w", newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=field)
        writer.writeheader()

def update_model(loss_function, optimizer, output, target, retain_graph=False):
    loss = loss_function(output, target)
    optimizer.zero_grad()
    loss.backward(retain_graph=retain_graph)
    optimizer.step()
    return loss

############ train model #############
for name in sitenames:
    sitename = name 
    print(sitename)
    train_dataset = PMSingleSiteDataset(sitename=sitename, target_hour=8, target_length=8, isTrain=True)
    valid_dataset = PMSingleSiteDataset(sitename=sitename, target_hour=8, target_length=8, isTrain=False)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    model = Seq2Seq(
                input_dim=15,
                emb_dim=64,
                output_dim=1,
                hid_dim=64,
                device=device,
                dropout=0.6,
                bidirectional=True,
            )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss(pos_weight=opt.pos_weight)

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
            x, y, ext = data 
            x = x.to(device) 
            y = y.to(device)
            ext = ext.to(device)
            # get loss & update
            o = model(x, y, ext)
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
            o = model.interface(x)
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
                os.rename("checkpoint.pt", os.path.join(cpt_dir, f"{sitename}.pt"))
                # write log
                with open(log_file, "a", newline='') as fp:
                    writer = csv.DictWriter(fp, fieldnames=field)
                    writer.writerow({
                        "sitename": sitename,
                        "best_loss": f"{best_loss:.3f}",
                        "epoch": epoch,
                        "timestamp": datetime.now()
                    })
                    #fp.write(f">>sitename: {sitename}\nepoch: {epoch}\nbest_mse_loss: {best_loss:.3f}\ntimestamp: {datetime.now()}\n\n")
                break
