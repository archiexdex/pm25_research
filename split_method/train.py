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
    if opt.method in ["all", "merged"]:
        train_dataset = PMDataset(sitename=sitename, config=opt, isTrain=True)
        valid_dataset = PMDataset(sitename=sitename, config=opt, isTrain=False)
    elif opt.method == "extreme":
        train_dataset = PMExtDataset(sitename=sitename, config=opt, use_ext=True, isTrain=True)
        valid_dataset = PMExtDataset(sitename=sitename, config=opt, use_ext=True, isTrain=False)
    elif opt.method == "normal":
        train_dataset = PMExtDataset(sitename=sitename, config=opt, use_ext=False, isTrain=True)
        valid_dataset = PMExtDataset(sitename=sitename, config=opt, use_ext=False, isTrain=False)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)
    
    # Model
    if opt.method == "merged":
        ext_model = load_model(os.path.join(cpt_dir, f"{sitename}_extreme.pt"), opt.model, opt).to(device)
        nor_model = load_model(os.path.join(cpt_dir, f"{sitename}_normal.pt"),  opt.model, opt).to(device)
        # MARK: - learnable model or fronzen?
        ext_model.eval()
        nor_model.eval()
        
        model = DNN_merged(
            ext_model=ext_model, 
            nor_model=nor_model,
            input_dim=opt.input_dim,
            output_dim=opt.output_dim,
            source_size=opt.source_size
        ).to(device)
    else:
        model = get_model(opt.model, opt).to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    ext_loss = EXTLoss()
    # criterion = ext_loss if opt.method == "extreme" else mse
    # criterion = mse
    criterion = ext_loss

    total_epoch = opt.total_epoch
    patience = opt.patience
    best_rmse = 1e9
    best_loss = 1e9
    earlystop_counter = 0
    st_time = datetime.now()
    
    for epoch in range(total_epoch):
        train_loss = train(model, train_dataloader, criterion, optimizer)
        valid_loss = test(model, valid_dataloader, criterion)
        if best_loss > valid_loss["loss"] or best_rmse > valid_loss["rmse"]:
            best_loss = min(best_loss, valid_loss["loss"])
            best_rmse = min(best_rmse, valid_loss["rmse"])
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
