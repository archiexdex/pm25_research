import os, shutil
import numpy as np
import random
import argparse
from argparse import Namespace
import json
import torch
from torch import nn
from sklearn.metrics import f1_score, precision_score
from model import *
from tqdm import tqdm
import csv
#torch.autograd.set_detect_anomaly(True)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, criterion, optimizer):
    model.train()
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
#             ext_loss  = bce(ext_pred, ext_true)
        pred_loss = criterion(y_pred, y_true)
        mse_loss  = mse(y_pred * thres_y, y_true * thres_y)
        loss = pred_loss
        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Record loss
#             mean_ext_loss  += ext_loss.item()
        mean_pred_loss += pred_loss.item()
        mean_rmse_loss += (torch.sqrt(mse_loss)).item()
        trange.set_description(\
            f"Training mean loss rmse: {mean_rmse_loss / (idx+1):.3f}, pred: {mean_pred_loss / (idx+1):.3e}")
    train_loss = mean_rmse_loss / len(dataloader)
    return train_loss

def test(model, dataloader, criterion):
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
        mse_loss  = mse(y_pred * thres_y, y_true * thres_y)
        # Record loss
        mean_rmse_loss += (torch.sqrt(mse_loss)).item()
        trange.set_description(f"Validation mean rmse: {mean_rmse_loss / (idx+1):.3f}")
    valid_loss = mean_rmse_loss / len(dataloader)
    return valid_loss

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

def get_path(path, no=None, mode=1):
    if no is not None:
        path = os.path.join(path, str(no))
    # Check whether path exists
    if mode == 1 and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def save_config(config, path, no, method):
    path = get_path(path, mode=0)
    path = os.path.join(path, f"{no}_{method}.json")
    config = vars(config)
    with open(path, 'w') as fp:
        json.dump(config, fp, ensure_ascii=False, indent=4)

def get_score(y_true, y_pred):
    f1       = f1_score(y_true, y_pred)
    macro    = f1_score(y_true, y_pred, average='macro')
    micro    = f1_score(y_true, y_pred, average='micro')
    weighted = f1_score(y_true, y_pred, average='weighted')
    return f1, macro, micro, weighted

def load_model(path, name, opt):
    checkpoint = torch.load(path)
    model = get_model(name, opt)
    model.load_state_dict(checkpoint)
    return model

def get_model(name, opt):
    if name == "dnn":
        model = DNN(
            input_dim=opt.input_dim, 
            emb_dim=opt.emb_dim, 
            hid_dim=opt.hid_dim, 
            output_dim=opt.output_dim,
            source_size=opt.source_size
        )
    elif name == "gru":
        model = GRU(
            input_dim     = opt.input_dim, 
            emb_dim       = opt.emb_dim, 
            hid_dim       = opt.hid_dim, 
            output_dim    = opt.output_dim,
            source_size   = opt.source_size,
            dropout       = opt.dropout,
            num_layers    = opt.num_layers,
            bidirectional = opt.bidirectional
        )
    #if name == "fudan":
    #    model = Fudan(
    #                input_dim=opt.input_dim,
    #                emb_dim=opt.emb_dim,
    #                output_dim=opt.output_dim,
    #                hid_dim=opt.hid_dim,
    #                device=device,
    #                dropout=opt.dropout,
    #                bidirectional=opt.bidirectional,
    #            )
    #elif name == "seq2seq":
    #    model = Seq2Seq(
    #                input_dim=opt.input_dim,
    #                emb_dim=opt.emb_dim,
    #                output_dim=opt.output_dim,
    #                hid_dim=opt.hid_dim,
    #                device=device,
    #                dropout=opt.dropout,
    #                bidirectional=opt.bidirectional,
    #            )
    #elif name == "cnn":
    #    model = CNNModel(
    #                input_dim=opt.input_dim,
    #                emb_dim=opt.emb_dim,
    #                output_dim=opt.output_dim,
    #                seq_len=opt.memory_size+opt.source_size,
    #                trg_len=1,
    #                device=device,
    #                dropout=opt.dropout,
    #            )
    #elif name == 'unet':
    #    model = UNET(
    #                input_dim=opt.input_dim,
    #                emb_dim=opt.emb_dim,
    #                output_dim=opt.output_dim,
    #                seq_len=opt.memory_size+opt.source_size,
    #                trg_len=1,
    #                device=device,
    #                dropout=opt.dropout,
    #            )
    #elif name == 'gru':
    #    model = SimpleGRU(
    #                input_dim=opt.input_dim,
    #                emb_dim=opt.emb_dim,
    #                output_dim=opt.output_dim,
    #                hid_dim=opt.hid_dim,
    #                target_length=opt.target_size,
    #            )
    #elif name == 'lstm':
    #    model = SimpleLSTM(
    #                input_dim=opt.input_dim,
    #                emb_dim=opt.emb_dim,
    #                output_dim=opt.output_dim,
    #                hid_dim=opt.hid_dim,
    #                target_length=opt.target_size,
    #            )
    #elif name == 'dnn':
    #    model = SimpleDNN(
    #                input_dim=opt.input_dim,
    #                emb_dim=opt.emb_dim,
    #                output_dim=opt.output_dim,
    #                hid_dim=opt.hid_dim,
    #                target_length=opt.target_size,
    #            )
    return model
