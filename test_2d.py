from utils import *
from tqdm import tqdm
import os, shutil 
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import DataLoader 
from constants import * 
from model import * 
from dataset import * 
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
opt = parse()
same_seeds(opt.seed)

if opt.no is not None:
    no = opt.no 
else: 
    print("n is not a number")
    exit() 

model_name = opt.model
cpt_dir = get_path(opt.cpt_dir, f"{no}", mode=0)
save_dir = opt.test_results_dir 
save_dir = get_path(save_dir, f"{no}")

if not os.path.exists(cpt_dir):
    print("Are you kidding me??? (,,ﾟДﾟ)")
    exit() 

mean_dict = {}
std_dict = {}
with open(opt.mean_path, "r") as fp:
    mean_dict = json.load(fp)
with open(opt.std_path, "r") as fp:
    std_dict = json.load(fp)

name_list = []
data_list = {"f1": [], "micro": [], "macro": [], "weighted": []}
valid_dataset = PMMultiSiteDataset(sitenames=sample_sites, config=opt, isTrain=False)
valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)

# Load mean, std
mean = np.array([mean_dict[s] for s in sample_sites])[:, 7].reshape(1, len(sample_sites), 1)
std  = np.array([ std_dict[s] for s in sample_sites])[:, 7].reshape(1, len(sample_sites), 1)

# Load model
model = CNN2DModel(
    input_dim=opt.input_dim,
    emb_dim=opt.emb_dim,
    output_dim=opt.output_dim,
    seq_len=opt.memory_size+opt.source_size,
    trg_len=1,
    device=device,
    dropout=opt.dropout,
)
checkpoint = torch.load(os.path.join(cpt_dir, f"{no}.pt"))
model.load_state_dict(checkpoint) 
model.to(device)
criterion = nn.MSELoss()

predict_list = None 
true_list = None
pred_list = None
sum_loss = 0
model.eval()
trange = tqdm(valid_dataloader)
for idx, data in enumerate(trange):
    # get data
    x, y, y_ext, past_window, past_ext, y_thres = map(lambda z: z.to(device), data)
    # get loss & update
    output = model(x, past_window, past_ext)
    loss = criterion(output, y)
    sum_loss += loss.item()
    # Denorm the data
    y = y.cpu().numpy() * std + mean
    output = output.cpu().detach().numpy() * std + mean 
    y_ext = y_ext.cpu().numpy()
    y_thres = y_thres.cpu().numpy()
    y_pred = np.zeros(output.shape)
    y_pred[output > y_thres] = 1
    y_pred[output <= y_thres] = 0
    # append result
    if predict_list is None:
        predict_list = output
        true_list = y_ext
        pred_list = y_pred
    else:
        predict_list = np.concatenate((predict_list, output), axis=0)
        true_list   = np.concatenate((true_list, y_ext), axis=0)
        pred_list   = np.concatenate((pred_list, y_pred), axis=0)
    trange.set_description(f"testing mean loss: {sum_loss / (idx+1):.4f}")

# summery value
valid_loss = sum_loss / len(valid_dataloader) 
#true_list = np.squeeze(true_list)
#pred_list = np.squeeze(pred_list)

# Evaluate each site
for idx, name in enumerate(sample_sites):
    sitename = name 
    print(f"sitename: {name}")

    f1, macro, micro, weighted = get_score(true_list[:, 0, idx, 0], pred_list[:, 0, idx, 0])
    name_list.append(sitename)
    data_list["f1"].append(f1); data_list["macro"].append(macro); data_list["micro"].append(micro); data_list["weighted"].append(weighted)
    #print(f"f1: {f1}, macro: {macro}, micro: {micro}, weighted: {weighted}")
    # Save results
    np.save(f"{save_dir}/{sitename}.npy", predict_list[:, :, idx, :])
    # save quantitative analysis
    df = pd.DataFrame({
        "sitename": name_list, 
        "f1":       data_list["f1"],
        "micro":    data_list["micro"],
        "macro":    data_list["macro"],
        "weighted": data_list["weighted"]
    })
    df.to_csv(f"{save_dir}/{no}.csv", index=False, encoding='utf_8_sig')