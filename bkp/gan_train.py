from utils import *
from custom_loss import *
from constants import *
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import PMExtDataset, PMDataset
import csv

# Train
opt = parse()
same_seeds(opt.seed)
save_config(opt, opt.config_dir, str(opt.no), opt.method)

if opt.no is not None:
    no = opt.no
else:
    print("n is not a number")
    exit()

cpt_dir = get_path(opt.cpt_dir, mode=0)
cpt_dir = get_path(cpt_dir, no, mode=0)
log_dir = get_path(opt.log_dir, mode=0)

device = get_device()

train_records = {}
for sitename in sitenames:
    if opt.skip_site == 1 and sitename not in sample_sites:
        continue
    print(sitename)
    
    # Dataset
    train_dataset = PMDataset(sitename=sitename, config=opt, isTrain=True)
    valid_dataset = PMDataset(sitename=sitename, config=opt, isTrain=False)
    # DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)
    # Model
    if opt.model in ["dnn"]:
        G = G_DNN(
                input_dim=opt.input_dim, 
                emb_dim=opt.emb_dim, 
                hid_dim=opt.hid_dim, 
                output_dim=opt.output_dim,
                source_size=opt.source_size
            ).to(device)
        D = D_DNN(
                input_dim=opt.output_dim, 
                emb_dim=opt.emb_dim, 
                hid_dim=opt.hid_dim, 
                output_dim=opt.output_dim,
                source_size=opt.target_size
            ).to(device)
    elif opt.model in ["gru"]:
        G = G_GRU(
            input_dim     = opt.input_dim, 
            emb_dim       = opt.emb_dim, 
            hid_dim       = opt.hid_dim, 
            output_dim    = opt.output_dim,
            source_size   = opt.source_size,
            dropout       = opt.dropout,
            num_layers    = opt.num_layers,
            bidirectional = opt.bidirectional
        ).to(device)
        D = D_GRU(
            input_dim     = opt.output_dim, 
            emb_dim       = opt.emb_dim, 
            hid_dim       = opt.hid_dim, 
            output_dim    = opt.output_dim,
            source_size   = opt.target_size,
            dropout       = opt.dropout,
            num_layers    = opt.num_layers,
            bidirectional = opt.bidirectional
        ).to(device)
        

    optim_g = optim.Adam(G.parameters(), lr=opt.lr)
    optim_d = optim.Adam(D.parameters(), lr=opt.lr)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    ext_loss = EXTLoss()

    total_epoch = opt.total_epoch
    patience = opt.patience
    best_rmse = 1e9
    best_loss = 1e9
    earlystop_counter = 0
    st_time = datetime.now()
    
    for epoch in range(total_epoch):
        train_loss = train_gan(G, D, train_dataloader, optim_g, optim_d)
        valid_loss = test_gan(G, valid_dataloader)
        if best_loss > valid_loss["loss"] or best_rmse > valid_loss["rmse"]:
            best_loss = min(best_loss, valid_loss["loss"])
            best_rmse = min(best_rmse, valid_loss["rmse"])
            torch.save(G.state_dict(), os.path.join(cpt_dir, f"{sitename}_{opt.method}.pt"))
            earlystop_counter = 0
            print(f">> Model saved epoch: {epoch}!!")

        # if best_loss doesn't improve for patience times, terminate training
        else:
            earlystop_counter += 1
            if earlystop_counter >= patience:
                print("Early stop!!!")
                break
    print(f"sitename: {sitename}\nepoch: {epoch}\nbest_rmse: {best_rmse: .4f}")
    train_records[sitename] = {
        "mode": opt.method,
        "best_rmse": f"{valid_loss['rmse']:.3f}", 
        "epoch": epoch, 
        "timestamp": datetime.now() - st_time
    }
# Write Record
write_record(f"{opt.log_dir}/{no}_{opt.method}.csv", train_records)

print("Done!!!")
