import os, shutil
import numpy as np
import random
import argparse
from argparse import Namespace
import json
import torch
from sklearn.metrics import f1_score, precision_score
torch.autograd.set_detect_anomaly(True)

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

def save_config(config, path, no):
    path = get_path(path, mode=0)
    path = os.path.join(path, f"{no}.json")
    config = vars(config)
    with open(path, 'w') as fp:
        json.dump(config, fp, ensure_ascii=False, indent=4)

def get_score(y_true, y_pred):
    f1       = f1_score(y_true, y_pred)
    macro    = f1_score(y_true, y_pred, average='macro')
    micro    = f1_score(y_true, y_pred, average='micro')
    weighted = f1_score(y_true, y_pred, average='weighted')
    return f1, macro, micro, weighted

def get_model(name, opt):
    if name == "fudan":
        model = Fudan(
                    input_dim=opt.input_dim,
                    emb_dim=opt.emb_dim,
                    output_dim=opt.output_dim,
                    hid_dim=opt.hid_dim,
                    device=device,
                    dropout=opt.dropout,
                    bidirectional=opt.bidirectional,
                )
    elif name == "seq2seq":
        model = Seq2Seq(
                    input_dim=opt.input_dim,
                    emb_dim=opt.emb_dim,
                    output_dim=opt.output_dim,
                    hid_dim=opt.hid_dim,
                    device=device,
                    dropout=opt.dropout,
                    bidirectional=opt.bidirectional,
                )
    elif name == "cnn":
        model = CNNModel(
                    input_dim=opt.input_dim,
                    emb_dim=opt.emb_dim,
                    output_dim=opt.output_dim,
                    seq_len=opt.memory_size+opt.source_size,
                    trg_len=1,
                    device=device,
                    dropout=opt.dropout,
                )
    elif name == 'unet':
        model = UNET(
                    input_dim=opt.input_dim,
                    emb_dim=opt.emb_dim,
                    output_dim=opt.output_dim,
                    seq_len=opt.memory_size+opt.source_size,
                    trg_len=1,
                    device=device,
                    dropout=opt.dropout,
                )
    elif name == 'gru':
        model = SimpleGRU(
                    input_dim=opt.input_dim,
                    emb_dim=opt.emb_dim,
                    output_dim=opt.output_dim,
                    hid_dim=opt.hid_dim,
                    target_length=opt.target_size,
                )
    elif name == 'lstm':
        model = SimpleLSTM(
                    input_dim=opt.input_dim,
                    emb_dim=opt.emb_dim,
                    output_dim=opt.output_dim,
                    hid_dim=opt.hid_dim,
                    target_length=opt.target_size,
                )
    elif name == 'dnn':
        model = SimpleDNN(
                    input_dim=opt.input_dim,
                    emb_dim=opt.emb_dim,
                    output_dim=opt.output_dim,
                    hid_dim=opt.hid_dim,
                    target_length=opt.target_size,
                )
    return model
