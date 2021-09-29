from utils import *
from constants import *
import itertools
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import *
from model import *
import csv

opt = parse()
opt.method = 'transformer'
opt.model  = 'transformer'
same_seeds(opt.seed)

device = get_device()
opt.device = device

check_train_id(opt)
build_dirs(opt)

save_config(opt)

st_t = datetime.now()
train_records = {}
for sitename in SITENAMES:
    if opt.skip_site == 1 and sitename not in SAMPLE_SITES:
        continue
    print(sitename)
    
    # Dataset
    train_dataset = PMDataset(sitename=sitename, opt=opt, isTrain=True)
    valid_dataset = PMDataset(sitename=sitename, opt=opt, isTrain=False)

    # Get ratio
    opt.ratio = train_dataset.get_ratio()
    print(f"Extreme Event: {1-opt.ratio:.3%}, Normal Event: {opt.ratio:.3%}")

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)
    
    # Model
    model = Transformer(opt, device).to(device)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # Parameters
    total_epoch = opt.total_epoch
    patience = opt.patience
    best_loss = 1e9
    earlystop_counter = 0
    st_time = datetime.now()
    
    for epoch in range(total_epoch):
        train_loss = tf_train(opt, train_dataloader, model, optimizer, device)
        valid_loss = tf_test (opt, valid_dataloader, model, device)
        if best_loss > valid_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(cpt_dir, f"{sitename}_{opt.method}.cpt"))
            earlystop_counter = 0
            print(f">> Model saved epoch: {epoch}!!")

        # if best_loss doesn't improve for patience times, terminate training
        else:
            earlystop_counter += 1
            if patience > 0 and earlystop_counter >= patience:
                print("Early stop!!!")
                break
    print(f"sitename: {sitename}\nepoch: {epoch}\nbest_loss: {best_loss: .4f}")

print(f"Finish training no: {no}, cost time: {datetime.now() - st_t}!!!")