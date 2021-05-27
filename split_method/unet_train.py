from utils import *
from custom_loss import *
from constants import *
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import *
import csv
from model import *

def train(model, dataloader, optimizer):
    model.train()
    mean_ext_loss = 0
    mean_pred_loss = 0
    mean_rmse_loss = 0
    trange = tqdm(dataloader)
    device = get_device()
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    fudan = FudanLoss()
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, thres_y = map(lambda z: z.to(device), data)
        # get loss & update
        y_pred = model(x)
        # Calculate loss
        pred_loss = mse(y_pred, y_true)
        loss = pred_loss
        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Get true value
        mse_loss = mse(y_true * thres_y, y_pred * thres_y)
        # Record loss
        mean_pred_loss += pred_loss.item()
        mean_rmse_loss += (torch.sqrt(mse_loss)).item()
        trange.set_description(\
            f"Training mean loss rmse: {mean_rmse_loss / (idx+1):.3e}, pred: {mean_pred_loss / (idx+1):.4e}")
    train_loss = mean_pred_loss / len(dataloader)
    return train_loss

def test(model, dataloader):
    # Validation
    model.eval()
    mean_ext_loss = 0
    mean_pred_loss = 0
    mean_rmse_loss = 0
    trange = tqdm(dataloader)
    device = get_device()
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    fudan = FudanLoss()
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, thres_y = map(lambda z: z.to(device), data)
        # get loss & update
        y_pred = model(x)
        # Calculate loss
        pred_loss = mse(y_pred, y_true)
        # Get true value
        mse_loss = mse(y_true * thres_y, y_pred * thres_y)
        # Record loss
        mean_pred_loss += pred_loss.item()
        mean_rmse_loss += (torch.sqrt(mse_loss)).item()
        trange.set_description(f"Validation mean rmse: \33[91m>>{mean_rmse_loss / (idx+1):.3e}<<\33[0m, pred: {mean_pred_loss / (idx+1):.4e}")
    valid_loss = mean_pred_loss / len(dataloader)
    mean_rmse_loss = mean_rmse_loss / len(dataloader)
    return {
        "loss" : valid_loss,
        "rmse": mean_rmse_loss
    }

        
if __name__ == "__main__":
    # Train
    opt = parse()
    opt.method = 'unet'
    same_seeds(opt.seed)
    save_config(opt, opt.config_dir, str(opt.no), opt.method)
    sitenames = sample_sites if opt.skip_site else sitenames

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
        print(sitename)
        
        # Dataset
        train_dataset = PMUnetDataset(sitename=sitename, config=opt, isTrain=True )
        valid_dataset = PMUnetDataset(sitename=sitename, config=opt, isTrain=False)
        # DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)
        # Model
        model = UNet_1d(
            c_in=opt.input_dim, 
            c_hid=opt.hid_dim, 
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
        # Parameters
        total_epoch = opt.total_epoch
        patience = opt.patience
        best_rmse = 1e9
        best_loss = 1e9
        earlystop_counter = 0
        st_time = datetime.now()
        
        for epoch in range(total_epoch):
            train_loss = train(model, train_dataloader, optimizer)
            valid_loss = test(model, valid_dataloader)
            if best_rmse > valid_loss["rmse"]:
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
        print(f"sitename: {sitename}\nepoch: {epoch}\nbest_loss: {best_rmse: .3e}")
        train_records[sitename] = {
            "mode": opt.method,
            "best_rmse": f"{best_rmse:.3e}", 
            "epoch": epoch, 
            "timestamp": datetime.now() - st_time
        }
    # Write Record
    write_record(f"{opt.log_dir}/{no}_{opt.method}.csv", train_records)

    print(f"Done training no: {no}!!!")
