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

name_list = []
data_list = {"f1": [], "micro": [], "macro": [], "weighted": []}
results = []
for sitename in SITENAMES:
    if opt.skip_site and sitename not in SAMPLE_SITES:
        continue
    print(sitename)
    dataset    = PMDataset(sitename=sitename, opt=opt, isTrain=False)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)
    
    # Model
    model = get_model(opt)
    # Load checkpoint
    model.load_state_dict(torch.load(os.path.join(cpt_dir, f"{sitename}_{method}.cpt")))
    # For device
    model.to(opt.device)
    # Freeze model
    model.eval()
    # Parameters
    #loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([95/5])).to(opt.device)
    loss_fn = nn.BCEWithLogitsLoss().to(opt.device)
    st_time = datetime.now()
    mean_loss = 0
    pred_list = None
    true_list = None
    trange = tqdm(dataloader)
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, past_data = map(lambda z: z.to(opt.device), data)
        # get loss & update
        if opt.model == "seq":
            _, y_pred = model(x, past_data)
        else:
            _, _, _, ext_pred = model(x, past_data)
        
        # Calculate loss
        loss = loss_fn(ext_pred, ext_true)
        # Record loss
        mean_loss += loss.item()
        trange.set_description(f"Test mean: {mean_loss / (idx+1):.3f}")
        # Recover predict
        ext_pred[ext_pred>=0.5] = 1
        ext_pred[ext_pred<0.5]  = 0
        ext_pred = ext_pred.detach().cpu().numpy()
        ext_true = ext_true.detach().cpu().numpy()
        # Append result
        if pred_list is None:
            pred_list = ext_pred
            true_list = ext_true
        else:
            pred_list = np.concatenate((pred_list, ext_pred), axis=0)
            true_list = np.concatenate((true_list, ext_true), axis=0)
    mean_loss /= len(dataloader)
    np.save(f"{rst_dir}/{sitename}.npy", pred_list)
    for j in [-1]:
        precision, recall, f1, macro, micro, weighted, mcc = get_score(true_list[:, j], pred_list[:, j])
        results.append({
            'sitename': sitename,
            'precision': f"{precision:.3f}",
            'recall'   : f"{recall   :.3f}",
            'f1'       : f"{f1       :.3f}",
            'mcc'      : f"{mcc      :.3f}",
            'macro'    : f"{macro    :.3f}",
            'micro'    : f"{micro    :.3f}",
            'weighted' : f"{weighted :.3f}",
        })
df = pd.DataFrame(results) 
df.to_csv(f"{rst_dir}/{no}_qa.csv", index=False, encoding='utf_8_sig')
print(f"Finish testing no: {no}, cost time: {datetime.now()-st_time}")
