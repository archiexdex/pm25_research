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
opt.model = 'fudan'
opt.method = 'fudan'
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
for sitename in SITENAMES:
    if opt.skip_site == 1 and sitename not in SAMPLE_SITES:
        continue
    print(sitename)
    
    # Dataset
    train_dataset = PMFudanDataset(sitename=sitename, config=opt, isTrain=True)
    valid_dataset = PMFudanDataset(sitename=sitename, config=opt, isTrain=False)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)
    
    # Model
    encoder = Fudan_Encoder(opt, device).to(device)
    history = Fudan_History(opt, device).to(device)
    decoder = Fudan_Decoder(opt, device).to(device)
    # Optimizer
    optimizer = optim.Adam(itertools.chain(encoder.parameters(), history.parameters(), decoder.parameters()), lr=opt.lr)
    # Parameters
    total_epoch = opt.total_epoch
    patience = opt.patience
    best_rmse = 1e9
    best_loss = 1e9
    earlystop_counter = 0
    st_time = datetime.now()
    
    for epoch in range(total_epoch):
        train_loss = fudan_train(opt, train_dataloader, encoder, history, decoder, optimizer, device)
        valid_loss = fudan_test (opt, valid_dataloader, encoder, history, decoder, device)
        if best_rmse > valid_loss:
            best_rmse = valid_loss
            torch.save(encoder.state_dict(), os.path.join(cpt_dir, f"{sitename}_{opt.method}_encoder.cpt"))
            torch.save(history.state_dict(), os.path.join(cpt_dir, f"{sitename}_{opt.method}_history.cpt"))
            torch.save(decoder.state_dict(), os.path.join(cpt_dir, f"{sitename}_{opt.method}_decoder.cpt"))
            earlystop_counter = 0
            print(f">> Model saved epoch: {epoch}!!")

        # if best_loss doesn't improve for patience times, terminate training
        else:
            earlystop_counter += 1
            if patience > 0 and earlystop_counter >= patience:
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
