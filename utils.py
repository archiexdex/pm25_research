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
        config = parser.parse_args(args=args)
    else:
        config = parser.parse_args()
    config = update_config(config)
    return config

def update_config(config):
    if config.is_concat_label:
        config.input_dim += 1
    return config

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
    elif name == "fudan":
        model = Fudan(opt)
    return model.to(device)

def get_mask(opt, data, thres_data):
    # Minimize the maximum threshold to opt.threshold.
    # Sometimes, it only influence winter threshold 
    _tmp = thres_data[:, 7]
    index = np.argwhere(_tmp>=opt.threshold)
    thres_data[index, 7] = opt.threshold
    # Calculate slope for adding extreme data
    # TODO: - use moving average to calculate dif_data
    dif_data = abs(data[1:, 7] - data[:-1, 7]) if opt.use_abs_delta else data[1:, 7] - data[:-1, 7]
    index = np.argwhere(dif_data>=opt.delta)[:, 0] + 1
    mask = np.zeros((data.shape[0], 1))
    mask[index] = 1
    index = np.argwhere(data[:, 7]>=thres_data[:, 7])
    mask[index] = 1
    return mask

