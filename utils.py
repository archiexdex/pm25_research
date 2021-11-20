import os, shutil
import numpy as np
import random
import argparse
from argparse import Namespace
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import *
from networks import *
from custom_loss import *
import csv
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef
#torch.autograd.set_detect_anomaly(True)

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def check_train_id(opt):
    assert opt.no != None, f"no should be a number"

def build_dirs(opt):
    try:
        cfg_dir = os.makedirs(os.path.join(opt.cfg_dir), 0o777)
    except:
        pass
    cpt_dir = os.path.join(opt.cpt_dir, str(opt.no))
    log_dir = os.path.join(opt.log_dir, str(opt.no))

    if (not opt.yes) and os.path.exists(cpt_dir):
        res = input(f"no: {no} exists, are you sure continue training? It will override all files.[y:N]")
        res = res.lower()
        assert res in ["y", "yes"], "Stop training!!"
        print("Override all files.")

    if os.path.exists(cpt_dir):
        shutil.rmtree(cpt_dir)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(cpt_dir, 0o777)
    os.makedirs(log_dir, 0o777)

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
    torch.backends.cudnn.enabled = False

#################################
########## config ###############
#################################
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
    if not config.no_concat_label:
        config.input_dim += 1
    return config

def save_config(config):
    no = config.no
    method = config.method
    _config = vars(config)
    with open(os.path.join(config.cfg_dir, f"{no}.json"), "w") as fp:
        json.dump(_config, fp, ensure_ascii=False, indent=4)

#################################
########## matrix ###############
#################################
def get_score(y_true, y_pred):
    precision = precision_score  (y_true, y_pred, zero_division=0)
    recall    = recall_score     (y_true, y_pred, zero_division=0)
    f1        = f1_score         (y_true, y_pred, zero_division=0)
    macro     = f1_score         (y_true, y_pred, zero_division=0, average='macro')
    micro     = f1_score         (y_true, y_pred, zero_division=0, average='micro')
    weighted  = f1_score         (y_true, y_pred, zero_division=0, average='weighted')
    mcc       = matthews_corrcoef(y_true, y_pred)
    return precision, recall, f1, macro, micro, weighted, mcc

#################################
########## model ################
#################################
def get_merged_model(opt, sitename):
    assert opt.nor_load_model != None, f"Merged method should determine the load model"
    assert opt.ext_load_model != None, f"Merged method should determine the load model"
    nor_load_path = os.path.join(opt.cpt_dir, str(opt.nor_load_model), f"{sitename}.cpt")
    ext_load_path = os.path.join(opt.cpt_dir, str(opt.ext_load_model), f"{sitename}.cpt")
    nor_model = load_model(nor_load_path, opt)
    ext_model = load_model(ext_load_path, opt)
    for p in nor_model.parameters():
        p.requires_grad = False
    for p in ext_model.parameters():
        p.requires_grad = False
    model = Merged_Model(opt, nor_model, ext_model)
    return model

def load_model(path, opt):
    checkpoint = torch.load(path)
    model = get_model(opt)
    model.load_state_dict(checkpoint)
    return model

def get_model(opt):
    name = opt.model.lower()
    if name == "dnn":
        model = DNN(opt)
    elif name == "rnn":
        model = RNN(opt)
    elif name == "seq":
        model = Seq2Seq(opt)
    elif name == "fudan":
        model = Fudan(opt)
    elif name == "transformer":
        model = Transformer(opt)
    return model

def get_loss(opt):
    if opt.loss == "bce":
        loss_fn = nn.BCELoss().to(opt.device)
    elif opt.loss == "evl":
        loss_fn = EVLoss(alpha=opt.ratio, gamma=opt.gamma).to(opt.device)
    return loss_fn

def get_trainer(opt):
    if opt.method == "fudan":
        trainer = fudan_trainer
    elif opt.method == "class":
        trainer = class_trainer
    elif opt.method == "transformer":
        trainer = tf_trainer
    elif opt.method == "merged":
        trainer = merged_trainer
    else:
        raise ValueError(f"--method does not support {opt.method}")
    return trainer

#################################
########## dataset ##############
#################################
def get_dataset(opt, sitename, isTrain):
    from dataset import PMDataset, PMFudanDataset
    if opt.method == "fudan":
        return PMFudanDataset(sitename=sitename, opt=opt, isTrain=isTrain)
    else:
        return PMDataset(sitename=sitename, opt=opt, isTrain=isTrain)

def read_file(sitename, opt, mode, isTrain):
    if mode == 0:
        read_path = os.path.join(opt.origin_train_dir, f"{sitename}.npy") if isTrain else os.path.join(opt.origin_valid_dir, f"{sitename}.npy")
    elif mode == 1:
        read_path = os.path.join(opt.thres_train_dir, f"{sitename}.npy") if isTrain else os.path.join(opt.thres_valid_dir, f"{sitename}.npy")
    if os.path.exists(read_path):
        data = np.load(read_path)
    else:
        raise ValueError(f"path {read_path} doesn't exist")
    return data

def get_mask(opt, data, thres_data):
    mask = np.zeros((data.shape[0], 1))
    # Limit the maximum threshold to opt.threshold.
    # Sometimes, it only influence winter threshold 
    if opt.use_threshold:
        if opt.use_min_threshold:
            index = np.argwhere(thres_data[:, 7] >= opt.threshold)
            thres_data[index, 7] = opt.threshold
        mask[data[:, 7]>=thres_data[:, 7]] = 1
    # Calculate slope for adding extreme data
    # TODO: - use moving average to calculate dif_data
    if opt.use_delta:
        dif_data = abs(data[1:, 7] - data[:-1, 7]) if opt.use_abs_delta else data[1:, 7] - data[:-1, 7]
        index = np.argwhere(dif_data>=opt.delta)[:, 0] + 1
        mask[index] = 1
    return mask

def get_split_dataset(opt, data, mask):
    """
        data: [data len, input_dim] 
        mask: [data len, 1]
        mode: 'norm', 'ext'
    """
    size = data.shape[0]
    shift = opt.memory_size + opt.source_size + opt.target_size
    _data = []
    _mask = []
    if opt.split_mode == "norm":
        for i in range(size - shift) :
            data_patch = data[i: i + shift]
            mask_patch = mask[i: i + shift]
            # check whether the extreme event is in the target_size? 
            if np.sum(mask_path[shift - opt.target_size:]) < 1:
                _data.append(data_path)
                _mask.append(mask_path)
    elif opt.split_mode == "ext":
        for i in range(size - shift) :
            data_path = data[i: i + shift]
            mask_path = mask[i: i + shift]
            # check whether the extreme event is in the target_size? 
            if np.sum(mask[shift - opt.target_size:]) > 0:
                _data.append(data_path)
                _mask.append(mask_path)

    return np.array(_data), np.array(_mask)
