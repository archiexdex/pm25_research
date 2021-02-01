import os, shutil
import numpy as np
import random
import argparse
from argparse import Namespace
import json
import torch
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

def parse():
    parser = argparse.ArgumentParser()
    try: 
        from argument import add_arguments 
        parser = add_arguments(parser)
    except:
        pass 
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
