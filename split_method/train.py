from utils import *
from constants import *
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
same_seeds(opt.seed)

if opt.no is not None:
    no = opt.no
else:
    print("n is not a number")
    exit()

try:
    cfg_dir = os.makedirs(os.path.join(opt.cfg_dir), 0o777)
except:
    pass
cpt_dir = os.path.join(opt.cpt_dir, str(no))
log_dir = os.path.join(opt.log_dir, str(no))
if (not opt.yes) and os.path.exists(cpt_dir):
    res = input(f"no: {no} exists, are you sure continue training? It will override all files.[y:N]")
    res = res.lower()
    if res not in ["y", "yes"]:
        print("Stop training")
        exit()
    print("Override all files.")
if os.path.exists(cpt_dir):
    shutil.rmtree(cpt_dir)
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(cpt_dir, 0o777)
os.makedirs(log_dir, 0o777)
save_config(opt)

device = get_device()

train_records = {}
for sitename in sitenames:
    if opt.skip_site == 1 and sitename not in sample_sites:
        continue
    print(sitename)
    
    # Dataset
    train_dataset = PMDataset(sitename=sitename, config=opt, isTrain=True)
    valid_dataset = PMDataset(sitename=sitename, config=opt, isTrain=False)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)
    
    # Model
    if opt.method == "merged":
        nor_model = load_model(os.path.join(f"checkpoints/88/{sitename}_normal.pt"),  opt).to(device)
        ext_model = load_model(os.path.join(f"checkpoints/89/{sitename}_extreme.pt"), opt).to(device)
        # MARK: - learnable model or fronzen?
        nor_model.eval()
        ext_model.eval()
        
        model = DNN_merged(
            opt=opt,
            nor_model=nor_model,
            ext_model=ext_model, 
        ).to(device)
    else:
        model = get_model(opt).to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    device = get_device()

    total_epoch = opt.total_epoch
    patience = opt.patience
    best_rmse = 1e9
    best_loss = 1e9
    earlystop_counter = 0
    st_time = datetime.now()
    
    for epoch in range(total_epoch):
        train_loss = train(opt, model, train_dataloader, optimizer, device)
        valid_loss = test (opt, model, valid_dataloader, device)
        if best_loss > valid_loss["loss"] or best_rmse > valid_loss["rmse"]:
            best_loss = valid_loss["loss"]
            best_rmse = valid_loss["rmse"]
            torch.save(model.state_dict(), os.path.join(cpt_dir, f"{sitename}_{opt.method}.pt"))
            earlystop_counter = 0
            print(f">> Model saved epoch: {epoch}!!")

        # if best_loss doesn't improve for patience times, terminate training
        else:
            earlystop_counter += 1
            if earlystop_counter >= patience:
                print("Early stop!!!")
                break
    print(f"sitename: {sitename}\nepoch: {epoch}\nbest_loss: {best_rmse: .4f}")
    train_records[sitename] = {
        "mode": opt.method,
        "best_rmse": f"{best_rmse:.3f}", 
        "epoch": epoch, 
        "timestamp": datetime.now() - st_time
    }
# Write Record
write_record(f"{opt.log_dir}/{no}_{opt.method}.csv", train_records)

print("Done!!!")
