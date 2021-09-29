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

def train_discrete(model, dataloader, optimizer):
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
        # Only consider the 8 hr data
        y_true = y_true[:, -1]
        thres_y = thres_y[:, -1]
        # get loss & update
        y_pred = model(x)
        # Calculate loss
        # pred_loss = bce(y_pred, y_true)
        pred_loss = fudan(y_pred, y_true)
        loss = pred_loss
        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Get true value
        true_value, true_index = torch.topk(y_true, 1, dim=1)
        pred_value, pred_index = torch.topk(y_pred, 1, dim=1)
        mse_loss = mse(true_index.float() * 10, pred_index.float() * 10)
        # Record loss
        mean_pred_loss += pred_loss.item()
        mean_rmse_loss += (torch.sqrt(mse_loss)).item()
        trange.set_description(\
            f"Training mean loss rmse: {mean_rmse_loss / (idx+1):.3f}, pred: {mean_pred_loss / (idx+1):.4e}")
    train_loss = mean_pred_loss / len(dataloader)
    return train_loss

def test_discrete(model, dataloader):
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
        y_true = y_true[:, -1]
        thres_y = thres_y[:, -1]
        # get loss & update
        y_pred = model(x)
        # Calculate loss
        # pred_loss = bce(y_pred, y_true)
        pred_loss = fudan(y_pred, y_true)
        # Get true value
        true_value, true_index = torch.topk(y_true, 1, dim=1)
        pred_value, pred_index = torch.topk(y_pred, 1, dim=1)
        mse_loss = mse(true_index.float() * 10, pred_index.float() * 10)
        # Record loss
        mean_pred_loss += pred_loss.item()
        mean_rmse_loss += (torch.sqrt(mse_loss)).item()
        trange.set_description(f"Validation mean rmse: \33[91m>>{mean_rmse_loss / (idx+1):.3f}<<\33[0m, pred: {mean_pred_loss / (idx+1):.4e}")
    valid_loss = mean_pred_loss / len(dataloader)
    mean_rmse_loss = mean_rmse_loss / len(dataloader)
    return {
        "loss" : valid_loss,
        "rmse": mean_rmse_loss
    }

        
if __name__ == "__main__":
    # Train
    opt = parse()
    opt.method = 'discrete'
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
        train_dataset = PMDiscreteDataset(sitename=sitename, config=opt, isTrain=True,  mode='normal')
        valid_dataset = PMDiscreteDataset(sitename=sitename, config=opt, isTrain=False, mode='normal')
        # DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)
        # Model
        if opt.model in ['dnn']: 
            model = Discrete_DNN(
                input_dim=opt.input_dim, 
                emb_dim=opt.emb_dim, 
                hid_dim=opt.hid_dim, 
                output_dim=21,
                source_size=opt.source_size
            ).to(device)
        elif opt.model in ['gru', 'lstm']:
            model = Discrete_GRU(
                input_dim=opt.input_dim, 
                emb_dim=opt.emb_dim, 
                hid_dim=opt.hid_dim, 
                output_dim=21,
                source_size=opt.source_size,
                num_layers=opt.num_layers,
                dropout=opt.dropout,
                bidirectional=opt.bidirectional,
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
            train_loss = train_discrete(model, train_dataloader, optimizer)
            valid_loss = test_discrete(model, valid_dataloader)
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
        # Dataset
        train_dataset = PMDiscreteDataset(sitename=sitename, config=opt, isTrain=True,  mode='ext')
        valid_dataset = PMDiscreteDataset(sitename=sitename, config=opt, isTrain=False, mode='ext')
        # DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        # Parameters
        total_epoch = opt.total_epoch
        patience = 5
        best_rmse = 1e9
        best_loss = 1e9
        earlystop_counter = 0
        st_time = datetime.now()
        
        for epoch in range(total_epoch):
            train_loss = train_discrete(model, train_dataloader, optimizer)
            valid_loss = test_discrete(model, valid_dataloader)
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

    print(f"Done training no: {no}!!!")
