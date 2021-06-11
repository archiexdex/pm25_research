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

def fudan_train(opt, dataloader, encoder, history, decoder, optimizer, device):
    encoder.train()
    history.train()
    decoder.train()
    mean_pred_loss = 0
    mean_rmse_loss = 0
    trange = tqdm(dataloader)
    mseLoss   = nn.MSELoss().to(device)
    fudanLoss = FudanLoss() .to(device)
    for idx, data in enumerate(trange):
        # get data
        xs, ys, exts, thres_ys, past_windows, past_exts = map(lambda z: z.to(device), data)
        
        # Tensor to store decoder outputs
        batch_size  = xs.shape[0]
        trg_size    = xs.shape[1]
        #window_indicators = torch.zeros(batch_size, past_window.shape[1], 1).to(self.device)
        ext_preds = torch.zeros(batch_size, trg_size, 1).to(device)
        outputs   = torch.zeros(batch_size, trg_size, 1).to(device)
        mean_l2_loss = 0
        for j in range(trg_size):
            x           = xs[:, j:j+1]
            past_window = past_windows[:, j]
            past_ext    = past_exts   [:, j]
            #print(x.shape, past_window.shape, past_ext.shape)
            # Get history window latent
            history_window = encoder(past_window, mode=0)
            window_ext     = history(history_window)
            # Update window_indicator
            l2_loss = fudanLoss(window_ext, past_ext)
            mean_l2_loss += l2_loss
            # Pass through data 
            latent, hidden = encoder(x, mode=1)
            output, ext_pred = decoder(latent, hidden, history_window, past_ext)
            # Store to buffer
            #window_indicators[:, i] = window_indicator
            #print(ext_pred.shape, output.shape)
            ext_preds[:, j] = ext_pred[:, 0]
            outputs[:, j]   = output[:, 0]
        # Calculate loss
        mse_loss = mseLoss(outputs, ys)
        ext_loss = fudanLoss(ext_preds, exts)
        mean_l2_loss /= trg_size
        total_loss = mse_loss + ext_loss + mean_l2_loss
        # Update model
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # Record loss
        with torch.no_grad():
            rmse_loss  = torch.sqrt(mseLoss(outputs * thres_ys, ys * thres_ys)) 
            mean_rmse_loss += rmse_loss.item()
        trange.set_description(\
            f"Training mean loss rmse: {mean_rmse_loss / (idx+1):.3f}, output mse: {mse_loss.item():.3e}, ext: {ext_loss.item():.3e}, l2: {mean_l2_loss:.3e}")
    mean_loss = mean_pred_loss / len(dataloader)
    return mean_loss

def fudan_test(opt, dataloader, encoder, history, decoder, device):
    # Validation
    encoder.eval()
    history.eval()
    decoder.eval()
    mean_pred_loss = 0
    mean_rmse_loss = 0
    trange = tqdm(dataloader)
    mseLoss = nn.MSELoss().to(device)
    for idx, data in enumerate(trange):
        # get data
        xs, ys, exts, thres_ys, past_windows, past_exts = map(lambda z: z.to(device), data)
        
        # Tensor to store decoder outputs
        batch_size  = xs.shape[0]
        trg_size    = xs.shape[1]
        #window_indicators = torch.zeros(batch_size, past_window.shape[1], 1).to(self.device)
        ext_preds = torch.zeros(batch_size, trg_size, 1).to(device)
        outputs   = torch.zeros(batch_size, trg_size, 1).to(device)
        for j in range(trg_size):
            x           = xs[:, j:j+1]
            past_window = past_windows[:, j]
            past_ext    = past_exts   [:, j]
            #print(x.shape, past_window.shape, past_ext.shape)
            # Get history window latent
            history_window = encoder(past_window, mode=0)
            window_ext     = history(history_window)
            # Pass through data 
            latent, hidden = encoder(x, mode=1)
            output, ext_pred = decoder(latent, hidden, history_window, past_ext)
            output[output<0] = 0
            # Store to buffer
            ext_preds[:, j] = ext_pred[:, 0]
            outputs[:, j]   = output[:, 0]
        # Record loss
        rmse_loss  = torch.sqrt(mseLoss(outputs * thres_ys, ys * thres_ys)) 
        mean_rmse_loss += rmse_loss.item()
        trange.set_description(f"Validation mean rmse: \33[91m>>{mean_rmse_loss / (idx+1):.3f}<<\33[0m")
    mean_rmse_loss = mean_rmse_loss / len(dataloader)
    return mean_rmse_loss
