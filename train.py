from utils import *
from tqdm import tqdm
import os, shutil 
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader 
from datetime import datetime 
from model import * 
from dataset.dataset import PMSingleSiteDataset 
from constants import *
import csv 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
opt = parse()
same_seeds(opt.seed)
save_config(opt, opt.config_dir, str(opt.no))

if opt.no is not None:
    no = opt.no 
else: 
    print("n is not a number")
    exit() 

model_name = opt.model
cpt_dir = get_path(opt.cpt_dir, mode=0)
cpt_dir = get_path(cpt_dir, no)
log_dir = get_path(opt.log_dir, mode=0)
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
    if sitename not in ["南投", "士林", "埔里", "關山"]:
        continue 
    print(sitename)
    train_dataset = PMSingleSiteDataset(sitename=sitename, config=opt, isTrain=True)
    valid_dataset = PMSingleSiteDataset(sitename=sitename, config=opt, isTrain=False)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)

    if model_name == "fudan":
        model = Model(
                    input_dim=opt.input_dim,
                    emb_dim=opt.emb_dim,
                    output_dim=opt.output_dim,
                    hid_dim=opt.hid_dim,
                    device=device,
                    dropout=opt.dropout,
                    bidirectional=opt.bidirectional,
                )
    elif model_name == "seq2seq":
        model = Seq2Seq(
                    input_dim=opt.input_dim,
                    emb_dim=opt.emb_dim,
                    output_dim=opt.output_dim,
                    hid_dim=opt.hid_dim,
                    device=device,
                    dropout=opt.dropout,
                    bidirectional=opt.bidirectional,
                )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    total_epoch = opt.total_epoch
    patience = opt.patience
    best_loss = 1e9
    earlystop_counter = 0

    for epoch in range(total_epoch):
        print(f">>Epoch: {epoch}\n")
        model.train()
        mean_past_ext_loss = 0
        mean_target_ext_loss = 0
        mean_prediction_loss = 0
        trange = tqdm(train_dataloader)
        for idx, data in enumerate(trange):
            # get data
            x, y, y_ext, past_window, past_ext = map(lambda z: z.to(device), data)
            # get loss & update
            if model_name == "fudan":
                window_indicator, indicator_output, prediction = model(x, past_window, past_ext)
                # Calculate loss
                past_ext_loss = bce(window_indicator, past_ext)
                target_ext_loss = bce(indicator_output, y_ext)
                prediction_loss = mse(prediction, y)
                loss = target_ext_loss + prediction_loss + past_ext_loss
            elif model_name == "seq2seq":
                prediction = model(x, y)
                # Calculate loss
                prediction_loss = mse(prediction, y)
                loss = prediction_loss
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Record loss
            if model_name == "fudan":
                mean_past_ext_loss += past_ext_loss.item()
                mean_target_ext_loss += target_ext_loss.item()
            mean_prediction_loss += prediction_loss.item()
            trange.set_description(f"Training mean loss past_ext: {mean_past_ext_loss / (idx+1):.3e}, target_ext: {mean_target_ext_loss / (idx+1):.4f}, prediction: {mean_prediction_loss / (idx+1):.3e}")
        train_loss = mean_prediction_loss / len(train_dataloader)

        mean_prediction_loss = 0
        model.eval()
        trange = tqdm(valid_dataloader)
        for idx, data in enumerate(trange):
            # get data
            x, y, y_ext, past_window, past_ext = map(lambda x: x.to(device), data)
            # get loss & update
            if model_name == "fudan":
                _, _, prediction = model(x, past_window, past_ext)
            elif model_name == "seq2seq":
                prediction = model.interface(x)
            # Calculate loss
            prediction_loss = mse(prediction, y)
            # Record loss
            mean_prediction_loss += prediction_loss.item()
            # Show current record
            trange.set_description(f"Validation mean loss prediction: {mean_prediction_loss / (idx+1):.6e}")
        valid_loss = mean_prediction_loss / len(valid_dataloader) 
        
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
                print(f"sitename: {sitename}\nepoch: {epoch}\nbest_loss: {best_loss:.6f}")
                os.rename("checkpoint.pt", os.path.join(cpt_dir, f"{sitename}.pt"))
                # write log
                with open(log_file, "a", newline='') as fp:
                    writer = csv.DictWriter(fp, fieldnames=field)
                    writer.writerow({
                        "sitename": sitename,
                        "best_loss": f"{best_loss:.6f}",
                        "epoch": epoch,
                        "timestamp": datetime.now()
                    })
                break
