from utils import *
from constants import *
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import *
import csv, json
import pandas as pd
from dotted.collection import DottedDict

# Test
opt = parse()
with open(f"{opt.cfg_dir}/{opt.no}.json", "r") as fp:
    opt = json.load(fp)
opt = Namespace(**opt)
same_seeds(opt.seed)

no = opt.no
method = opt.method

same_seeds(opt.seed)
cpt_dir = os.path.join(opt.cpt_dir, str(no))
rst_dir = os.path.join(opt.rst_dir, str(no))
if not os.path.exists(rst_dir):
    os.makedirs(rst_dir, 0o777)

device = get_device()

results = []
for sitename in SITENAMES:
    if opt.skip_site and sitename not in SAMPLE_SITES:
        continue
    print(sitename)
    test_dataset    = PMFudanDataset(sitename=sitename, opt=opt, isTrain=False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    
    # Model
    model = Fudan(opt)
    # Load checkpoint
    model.load_state_dict(torch.load(os.path.join(cpt_dir, f"{sitename}_{method}.cpt")))
    # For device
    model.to(device)
    # Freeze model
    model.eval()
    # Parameters
    mseLoss = nn.MSELoss().to(device)
    fudanLoss = FudanLoss().to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([95/5])).to(device)
    st_time = datetime.now()
    mean_rmse_loss = 0
    mean_pred_loss = 0
    true_list  = None
    pred_list  = None
    value_list = None
    trange = tqdm(test_dataloader)
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
        # Recover y
        recover_y = y_pred * thres_y
        ext_pred[ext_pred>= 0.5] = 1
        ext_pred[ext_pred<  0.5] = 0
        recover_y = recover_y.detach().cpu().numpy()
        ext_pred  = ext_pred .detach().cpu().numpy()
        ext_true  = ext_true .detach().cpu().numpy()
        # Append result
        if value_list is None:
            value_list = recover_y
            true_list  = ext_true
            pred_list  = ext_pred
        else:
            value_list = np.concatenate((value_list, recover_y), axis=0)
            true_list  = np.concatenate((true_list, ext_true),   axis=0)
            pred_list  = np.concatenate((pred_list, ext_pred),   axis=0)
        trange.set_description(f"Test mean rmse: {mean_rmse_loss / (idx+1):.3f} pred: {mean_pred_loss / (idx+1):.3f}")
    #test_loss = mean_rmse_loss / len(test_dataloader)
    
    # Save the prediction value
    np.save(f"{rst_dir}/{sitename}.npy", value_list)
    np.save(f"{rst_dir}/{sitename}_class.npy", pred_list)
    # Record quantitative index
    #precision, recall, f1, macro, micro, weighted = get_score(true_list, pred_list)
    #print(f"precision: {precision}, recall: {recall}, f1: {f1}, macro: {macro}, micro: {micro}, weighted: {weighted}")
    for j in [-1]:
        precision, recall, f1, macro, micro, weighted = get_score(true_list[:, j], pred_list[:, j])
        results.append({
            'sitename': sitename,
            'target': j,
            'precision': f"{precision:.3f}",
            'recall'   : f"{recall   :.3f}",
            'f1'       : f"{f1       :.3f}",
            'macro'    : f"{macro    :.3f}",
            'micro'    : f"{micro    :.3f}",
            'weighted' : f"{weighted :.3f}"
        })
df = pd.DataFrame(results) 
df.to_csv(f"{rst_dir}/{no}_qa.csv", index=False, encoding='utf_8_sig')
