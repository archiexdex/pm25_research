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

def train_gan(G, D, dataloader, optim_g, optim_d):
    G.train()
    D.train()
    mean_ext_loss = 0
    mean_pred_loss = 0
    mean_rmse_loss = 0
    trange = tqdm(dataloader)
    device = get_device()
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    ext = EXTLoss()
    for idx, data in enumerate(trange):
        
        # get data
        x, y_true, ext_true, thres_y = map(lambda z: z.to(device), data)

        # Adversarial ground truths
        valid = torch.ones(x.shape[0], 1).to(device)
        fake  = torch.zeros(x.shape[0], 1).to(device)

        # -----------------
        #  Train Generator
        # -----------------
        y_pred = G(x)
        
        real_d = D(y_true.detach())
        fake_d = D(y_pred)

        adv_loss = bce(fake_d - real_d.mean(0, keepdim=True), valid)
        content_loss = ext(y_pred, y_true)
        g_loss = adv_loss + content_loss

        optim_g.zero_grad()
        g_loss.backward()
        optim_g.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        real_d = D(y_true)
        fake_d = D(y_pred.detach())

        real_loss = bce(real_d - fake_d.mean(0, keepdim=True), valid)
        fake_loss = bce(fake_d - real_d.mean(0, keepdim=True), fake)

        d_loss = (real_loss + fake_loss) / 2

        optim_d.zero_grad()
        d_loss.backward()
        optim_d.step()

        # ---------------------
        #  Record Loss
        # ---------------------
        mse_loss  = mse(y_pred[:, -1] * thres_y[:, -1], y_true[:, -1] * thres_y[:, -1])
        mean_rmse_loss += (torch.sqrt(mse_loss)).item()
        trange.set_description(\
            f"Training mean loss rmse: {mean_rmse_loss / (idx+1):.3f}, g adv: {adv_loss.item():.4e}, d adv: {d_loss.item():.4e}")
    train_loss = mean_pred_loss / len(dataloader)
    return train_loss

def test_gan(model, dataloader):
    # Validation
    model.eval()
    mean_ext_loss = 0
    mean_pred_loss = 0
    mean_rmse_loss = 0
    trange = tqdm(dataloader)
    device = get_device()
    mse = nn.MSELoss()
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, thres_y = map(lambda z: z.to(device), data)
        y_true = y_true[:, -1]
        thres_y = thres_y[:, -1]
        # get loss & update
        y_pred = model(x)
        # Calculate loss
        mse_loss  = mse(y_pred[:, -1] * thres_y, y_true * thres_y)
        # Record loss
        mean_rmse_loss += (torch.sqrt(mse_loss)).item()
        trange.set_description(f"Validation mean rmse: \33[91m>>{mean_rmse_loss / (idx+1):.3f}<<\33[0m")
    valid_loss = mean_pred_loss / len(dataloader)
    mean_rmse_loss = mean_rmse_loss / len(dataloader)
    return {
        "loss" : valid_loss,
        "rmse": mean_rmse_loss
    }

def fudan_train(opt, dataloader, model, optimizer, device):
    model.train()
    mean_pred_loss = 0
    mean_rmse_loss = 0
    trange = tqdm(dataloader)
    mseLoss   = nn.MSELoss().to(device)
    fudanLoss = FudanLoss() .to(device)
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, thres_y, past_window, past_ext = map(lambda z: z.to(device), data)
        
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

def fudan_test(opt, dataloader, model, device):
    # Validation
    model.eval()
    mean_pred_loss = 0
    mean_rmse_loss = 0
    trange = tqdm(dataloader)
    mseLoss = nn.MSELoss().to(device)
    fudanLoss = FudanLoss() .to(device)
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, thres_y, past_window, past_ext = map(lambda z: z.to(device), data)
        
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

def class_train(opt, dataloader, model, optimizer, device):
    model.train()
    mean_loss = 0
    trange = tqdm(dataloader)
    if opt.loss == "bce":
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(95/5)).to(device)
        #loss_fn = nn.BCEWithLogitsLoss().to(device)
    elif opt.loss == "fudan":
        loss_fn = FudanLoss().to(device)
    for idx, data in enumerate(trange):
        # get data
        x, y_true, past_window, past_ext = map(lambda z: z.to(device), data)
        # get loss & update
        if opt.model.lower() == "seq":
            _, y_pred = model(x, past_window) 
        else:
            _, _, _, y_pred = model(x, past_window)
        
        # Calculate loss
        loss = loss_fn(y_pred, y_true)
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

def class_test(opt, dataloader, model, device):
    # Validation
    model.eval()
    mean_loss = 0
    trange = tqdm(dataloader)
    if opt.loss == "bce":
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(95/5)).to(device)
        #loss_fn = nn.BCEWithLogitsLoss().to(device)
    elif opt.loss == "fudan":
        loss_fn = FudanLoss().to(device)
    for idx, data in enumerate(trange):
        # get data
        x, y_true, past_window, past_ext = map(lambda z: z.to(device), data)
        # get loss & update
        if opt.model.lower() == "seq":
            _, y_pred = model(x, past_window) 
        else:
            _, _, _, y_pred = model(x, past_window)
        # Calculate loss
        loss = loss_fn(y_pred, y_true)
        # Record loss
        mean_loss += loss.item()
        trange.set_description(f"Validation mean: \33[91m>>{mean_loss / (idx+1):.3f}<<\33[0m")
    mean_loss /= len(dataloader)
    return  mean_loss

def sa_train(opt, dataloader, model, optimizer, device):
    model.train()
    mean_loss = 0
    trange = tqdm(dataloader)
    if opt.loss == "bce":
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(95/5)).to(device)
        #loss_fn = nn.BCEWithLogitsLoss().to(device)
    elif opt.loss == "fudan":
        loss_fn = FudanLoss().to(device)
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, past_data = map(lambda z: z.to(device), data)
        # get loss & update
        ext_pred, _ = model(past_data, x)
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

def sa_test(opt, dataloader, model, device):
    model.eval()
    mean_loss = 0
    trange = tqdm(dataloader)
    if opt.loss == "bce":
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(95/5)).to(device)
        #loss_fn = nn.BCEWithLogitsLoss().to(device)
    elif opt.loss == "fudan":
        loss_fn = FudanLoss().to(device)
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, past_data = map(lambda z: z.to(device), data)
        # get loss & update
        ext_pred, _ = model(past_data, x)
        # Calculate loss
        loss = loss_fn(ext_pred, ext_true)
        # Record loss
        mean_loss += loss.item()
        trange.set_description(f"Validation mean: \33[91m>>{mean_loss / (idx+1):.3f}<<\33[0m")
    mean_loss /= len(dataloader)
    return  mean_loss
