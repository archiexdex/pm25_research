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
from dataset.dataset import PMSingleSiteDataset 

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
with open("dataset/train_mean.json", "r") as fp:
    mean_dict = json.load(fp)
with open("dataset/train_std.json", "r") as fp:
    std_dict = json.load(fp)

for name in sitenames:
    sitename = name 
    if sitename not in ["南投", "士林", "埔里", "關山"]:
        continue 
    print(f"sitename: {name}")
    valid_dataset = PMSingleSiteDataset(sitename=sitename, config=opt, isTrain=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)

    mean = mean_dict[sitename][7]
    std = std_dict[sitename][7]

    # Load model
    if model_name == "fudan":
        model = Model(
                    input_dim=opt.input_dim,
                    emb_dim=opt.emb_dim,
                    output_dim=opt.output_dim,
                    hid_dim=opt.hid_dim,
                    device=device,
                    dropout=opt.dropout,
                    bidirectional=opt.bidirectional,
                )
    elif model_name == "seq2seq":
        model = Seq2Seq(
                    input_dim=opt.input_dim,
                    emb_dim=opt.emb_dim,
                    output_dim=opt.output_dim,
                    hid_dim=opt.hid_dim,
                    device=device,
                    dropout=opt.dropout,
                    bidirectional=opt.bidirectional,
                )
    checkpoint = torch.load(os.path.join(cpt_dir, f"{sitename}.pt"))
    model.load_state_dict(checkpoint) 
    model.to(device)
    criterion = nn.MSELoss()

    predict_result = np.array([])
    sum_loss = 0
    model.eval()
    trange = tqdm(valid_dataloader)
    for idx, data in enumerate(trange):
        # get data
        x, y, y_ext, past_window, past_ext = map(lambda z: z.to(device), data)
        # get loss & update
        if model_name == "fudan":
            _, _, output = model(x, past_window, past_ext)
        elif model_name == "seq2seq":
            output = model.interface(x)
        loss = criterion(output, y)
        sum_loss += loss.item()
        # Denorm the data
        predict_o = output.cpu().detach().numpy() * std + mean 
        # append result
        #predict_result.append(predict_o)
        predict_result = np.concatenate((predict_result, predict_o), axis=None)
        trange.set_description(f"testing mean loss: {sum_loss / (idx+1):.4f}")

    valid_loss = sum_loss / len(valid_dataloader) 
        
    # Save results
    np.save(f"{save_dir}/{sitename}.npy", predict_result)
