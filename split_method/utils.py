import os, shutil
import numpy as np
import random
import argparse
from argparse import Namespace
import json
import torch
from torch import nn
from sklearn.metrics import f1_score, precision_score, recall_score 
from networks import *
from tqdm import tqdm
import csv
torch.autograd.set_detect_anomaly(True)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_gan(G, D, dataloader, optim_g, optim_d):
    G.train()
    D.train()
    mean_ext_loss = 0
    mean_pred_loss = 0
    mean_rmse_loss = 0
    trange = tqdm(dataloader)
    device = get_device()
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    ext = EXTLoss()
    for idx, data in enumerate(trange):
        
        # get data
        x, y_true, ext_true, thres_y = map(lambda z: z.to(device), data)

        # Adversarial ground truths
        valid = torch.ones(x.shape[0], 1).to(device)
        fake  = torch.zeros(x.shape[0], 1).to(device)

        # -----------------
        #  Train Generator
        # -----------------
        y_pred = G(x)
        
        real_d = D(y_true.detach())
        fake_d = D(y_pred)

        adv_loss = bce(fake_d - real_d.mean(0, keepdim=True), valid)
        content_loss = ext(y_pred, y_true)
        g_loss = adv_loss + content_loss

        optim_g.zero_grad()
        g_loss.backward()
        optim_g.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        real_d = D(y_true)
        fake_d = D(y_pred.detach())

        real_loss = bce(real_d - fake_d.mean(0, keepdim=True), valid)
        fake_loss = bce(fake_d - real_d.mean(0, keepdim=True), fake)

        d_loss = (real_loss + fake_loss) / 2

        optim_d.zero_grad()
        d_loss.backward()
        optim_d.step()

        # ---------------------
        #  Record Loss
        # ---------------------
        mse_loss  = mse(y_pred[:, -1] * thres_y[:, -1], y_true[:, -1] * thres_y[:, -1])
        mean_rmse_loss += (torch.sqrt(mse_loss)).item()
        trange.set_description(\
            f"Training mean loss rmse: {mean_rmse_loss / (idx+1):.3f}, g adv: {adv_loss.item():.4e}, d adv: {d_loss.item():.4e}")
    train_loss = mean_pred_loss / len(dataloader)
    return train_loss

def test_gan(model, dataloader):
    # Validation
    model.eval()
    mean_ext_loss = 0
    mean_pred_loss = 0
    mean_rmse_loss = 0
    trange = tqdm(dataloader)
    device = get_device()
    mse = nn.MSELoss()
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, thres_y = map(lambda z: z.to(device), data)
        y_true = y_true[:, -1]
        thres_y = thres_y[:, -1]
        # get loss & update
        y_pred = model(x)
        # Calculate loss
        mse_loss  = mse(y_pred[:, -1] * thres_y, y_true * thres_y)
        # Record loss
        mean_rmse_loss += (torch.sqrt(mse_loss)).item()
        trange.set_description(f"Validation mean rmse: \33[91m>>{mean_rmse_loss / (idx+1):.3f}<<\33[0m")
    valid_loss = mean_pred_loss / len(dataloader)
    mean_rmse_loss = mean_rmse_loss / len(dataloader)
    return {
        "loss" : valid_loss,
        "rmse": mean_rmse_loss
    }

def write_record(path, records):
    header = ["sitename", "mode", "best_rmse", "epoch", "cost_time"]
    with open(path, "w") as fp:
        #json.dump(train_ext_records,  fp, ensure_ascii=False, indent=4)
        writer = csv.DictWriter(fp, fieldnames=header)
        writer.writeheader()
        for sitename in records:
            writer.writerow({
                "sitename": sitename,
                "mode":      records[sitename]["mode"],
                "best_rmse": records[sitename]["best_rmse"],
                "epoch":     records[sitename]["epoch"],
                "cost_time": records[sitename]["timestamp"]
            })

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse(args=None):
    parser = argparse.ArgumentParser()
    try: 
        from argument import add_arguments 
        parser = add_arguments(parser)
    except:
        pass 
    if args is not None:
        return parser.parse_args(args=args)
    else:
        return parser.parse_args()

def save_config(config):
    no = config.no
    method = config.method
    _config = vars(config)
    with open(os.path.join(config.cfg_dir, f"{no}_{method}.json"), "w") as fp:
        json.dump(_config, fp, ensure_ascii=False, indent=4)

def get_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score   (y_true, y_pred, zero_division=0)
    f1        = f1_score       (y_true, y_pred, zero_division=0)
    macro     = f1_score       (y_true, y_pred, zero_division=0, average='macro')
    micro     = f1_score       (y_true, y_pred, zero_division=0, average='micro')
    weighted  = f1_score       (y_true, y_pred, zero_division=0, average='weighted')
    return precision, recall, f1, macro, micro, weighted

def load_model(path, opt):
    checkpoint = torch.load(path)
    model = get_model(opt)
    model.load_state_dict(checkpoint)
    return model

def get_model(opt, device):
    name = opt.model.lower()
    if name == "dnn":
        model = DNN(opt)
    elif name == "gru":
        model = GRU(opt)
    elif name == "seq":
        model = Seq2Seq(opt, device)
    return model.to(device)
