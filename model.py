import torch
import torch.nn as nn
from tqdm import tqdm
from custom_loss import *

def train(opt, model, dataloader, optimizer, device):
    model.train()
    mean_pred_loss = 0
    mean_rmse_loss = 0
    trange = tqdm(dataloader)
    mseLoss   = nn.MSELoss()
    extLoss   = EXTLoss()
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
        elif opt.method in ["normal"]:
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
        elif opt.method in ["normal"]:
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

def fudan_train(opt, dataloader, model, optimizer):
    model.train()
    mean_pred_loss = 0
    mean_rmse_loss = 0
    trange = tqdm(dataloader)
    mseLoss   = nn.MSELoss().to(opt.device)
    fudanLoss = FudanLoss() .to(opt.device)
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, thres_y, past_window, past_ext = map(lambda z: z.to(opt.device), data)
        
        y_pred, ext_pred, past_pred = model(x, past_window, past_ext)

        # Calculate loss
        mse_loss = mseLoss(y_pred, y_true)
        ext_loss = fudanLoss(ext_pred, ext_true)
        his_loss = fudanLoss(past_pred, past_ext)
        loss = mse_loss + ext_loss + his_loss
        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Record loss
        with torch.no_grad():
            rmse_loss  = torch.sqrt(mseLoss(y_pred * thres_y, y_true * thres_y)) 
            mean_rmse_loss += rmse_loss.item()
            mean_pred_loss += ext_loss.item()
        trange.set_description(\
            f"Training mean loss rmse: {mean_rmse_loss / (idx+1):.3f}, pred: {mean_pred_loss / (idx+1):.3e}, output mse: {mse_loss.item():.3e}, ext: {ext_loss.item():.3e}, his: {his_loss:.3e}")
    mean_rmse_loss /= len(dataloader)
    mean_pred_loss /= len(dataloader)
    return mean_rmse_loss, mean_pred_loss

def fudan_test(opt, dataloader, model):
    # Validation
    model.eval()
    mean_pred_loss = 0
    mean_rmse_loss = 0
    trange = tqdm(dataloader)
    mseLoss = nn.MSELoss().to(opt.device)
    fudanLoss = FudanLoss() .to(opt.device)
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, thres_y, past_window, past_ext = map(lambda z: z.to(opt.device), data)
        
        y_pred, ext_pred, past_pred = model(x, past_window, past_ext)

        # Calculate loss
        mse_loss = mseLoss(y_pred, y_true)
        ext_loss = fudanLoss(ext_pred, ext_true)
        his_loss = fudanLoss(past_pred, past_ext)

        rmse_loss  = torch.sqrt(mseLoss(y_pred * thres_y, y_true * thres_y)) 
        mean_rmse_loss += rmse_loss.item()
        mean_pred_loss += ext_loss.item()
        trange.set_description(f"Validation mean rmse: \33[91m>>{mean_rmse_loss / (idx+1):.3f}, pred: {mean_pred_loss / (idx+1):.3e}<<\33[0m")
    mean_rmse_loss /= len(dataloader)
    mean_pred_loss /= len(dataloader)
    return mean_rmse_loss, mean_pred_loss

def class_train(opt, dataloader, model, optimizer):
    model.train()
    mean_loss = 0
    trange = tqdm(dataloader)
    if opt.loss == "bce":
        loss_fn = nn.BCEWithLogitsLoss().to(opt.device)
    elif opt.loss == "evl":
        loss_fn = EVLoss(alpha=opt.ratio, gamma=opt.gamma).to(opt.device)
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, past_data = map(lambda z: z.to(opt.device), data)
        # get loss & update
        if opt.model.lower() == "seq":
            _, ext_pred = model(x, past_data) 
        else:
            _, _, _, ext_pred = model(x, past_data)
        
        # Calculate loss
        loss = loss_fn(ext_pred, ext_true)
        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Record loss
        mean_loss += loss.item()
        trange.set_description(\
            f"Training mean loss: {mean_loss / (idx+1):.3f}")
    mean_loss /= len(dataloader)
    return mean_loss

def class_test(opt, dataloader, model):
    # Validation
    model.eval()
    mean_loss = 0
    trange = tqdm(dataloader)
    if opt.loss == "bce":
        loss_fn = nn.BCEWithLogitsLoss().to(opt.device)
    elif opt.loss == "evl":
        loss_fn = EVLoss(alpha=opt.ratio, gamma=opt.gamma).to(opt.device)
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, past_data = map(lambda z: z.to(opt.device), data)
        # get loss & update
        if opt.model.lower() == "seq":
            _, ext_pred = model(x, past_data) 
        else:
            _, _, _, ext_pred = model(x, past_data)
        # Calculate loss
        loss = loss_fn(ext_pred, ext_true)
        # Record loss
        mean_loss += loss.item()
        trange.set_description(f"Validation mean: \33[91m>>{mean_loss / (idx+1):.3f}<<\33[0m")
    mean_loss /= len(dataloader)
    return  mean_loss

def tf_train(opt, dataloader, model, optimizer):
    model.train()
    mean_loss = 0
    trange = tqdm(dataloader)
    if opt.loss == "bce":
        loss_fn = nn.BCEWithLogitsLoss().to(opt.device)
    elif opt.loss == "evl":
        loss_fn = EVLoss(alpha=opt.ratio, gamma=1).to(opt.device)
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, past_data = map(lambda z: z.to(opt.device), data)
        # get loss & update
        ext_pred, _, _ = model(past_data, x)
        # Calculate loss
        loss = loss_fn(ext_pred, ext_true)
        # Update model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
        optimizer.step()
        # Record loss
        mean_loss += loss.item()
        trange.set_description(\
            f"Training mean loss: {mean_loss / (idx+1):.3f}")
    mean_loss /= len(dataloader)
    return mean_loss

def tf_test(opt, dataloader, model):
    model.eval()
    mean_loss = 0
    trange = tqdm(dataloader)
    if opt.loss == "bce":
        loss_fn = nn.BCEWithLogitsLoss().to(opt.device)
    elif opt.loss == "evl":
        loss_fn = EVLoss(alpha=opt.ratio, gamma=1).to(opt.device)
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, past_data = map(lambda z: z.to(opt.device), data)
        # get loss & update
        ext_pred, _, _ = model(past_data, x)
        # Calculate loss
        loss = loss_fn(ext_pred, ext_true)
        # Record loss
        mean_loss += loss.item()
        trange.set_description(f"Validation mean: \33[91m>>{mean_loss / (idx+1):.3f}<<\33[0m")
    mean_loss /= len(dataloader)
    return  mean_loss
