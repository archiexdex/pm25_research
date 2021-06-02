import torch
import torch.nn as nn
from tqdm import tqdm
from custom_loss import *

def train(opt, model, dataloader, optimizer, device):
    model.train()
    mean_pred_loss = 0
    mean_rmse_loss = 0
    trange = tqdm(dataloader)
    mseLoss = nn.MSELoss()
    extLoss = EXTLoss()
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, thres_y = map(lambda z: z.to(device), data)
        y_true  = y_true 
        thres_y = thres_y
        #y_true  = y_true [:, -1]
        #thres_y = thres_y[:, -1]
        # get loss & update
        if opt.method == "merged":
            y_pred = model(x)
        else:
            _, _, _, y_pred = model(x)

        # Calculate loss
        if opt.method in ["extreme", 'merged']:
            pred_loss = extLoss(y_pred, y_true)
        else:
            pred_loss = mseLoss(y_pred, y_true)
        mse_loss  = mseLoss(y_pred * thres_y, y_true * thres_y)
        loss = pred_loss
        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Record loss
        mean_pred_loss += pred_loss.item()
        mean_rmse_loss += (torch.sqrt(mse_loss)).item()
        trange.set_description(\
            f"Training mean loss rmse: {mean_rmse_loss / (idx+1):.3f}, pred: {mean_pred_loss / (idx+1):.4e}")
    mean_loss = mean_pred_loss / len(dataloader)
    return mean_loss

def test(opt, model, dataloader, device):
    # Validation
    model.eval()
    mean_pred_loss = 0
    mean_rmse_loss = 0
    trange = tqdm(dataloader)
    mseLoss = nn.MSELoss()
    extLoss = EXTLoss()
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, thres_y = map(lambda z: z.to(device), data)
        y_true  = y_true 
        thres_y = thres_y
        #y_true  = y_true[:, -1]
        #thres_y = thres_y[:, -1]
        # get loss & update
        if opt.method == "merged":
            y_pred = model(x)
        else:
            _, _, _, y_pred = model(x)
        
        # Calculate loss
        if opt.method in ["extreme", 'merged']:
            pred_loss = extLoss(y_pred, y_true)
        else:
            pred_loss = mseLoss(y_pred, y_true)
        mse_loss  = mseLoss(y_pred * thres_y, y_true * thres_y)
        # Record loss
        mean_pred_loss += pred_loss.item()
        mean_rmse_loss += (torch.sqrt(mse_loss)).item()
        trange.set_description(f"Validation mean rmse: \33[91m>>{mean_rmse_loss / (idx+1):.3f}<<\33[0m, pred: {mean_pred_loss / (idx+1):.4e}")
    valid_loss     = mean_pred_loss / len(dataloader)
    mean_rmse_loss = mean_rmse_loss / len(dataloader)
    return {
        "loss" : valid_loss,
        "rmse": mean_rmse_loss
    }
