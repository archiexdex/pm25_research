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
st_time = datetime.now()

results = []
for sitename in SITENAMES:
    if opt.skip_site and sitename not in SAMPLE_SITES:
        continue
    print(sitename)
    # Dataset
    dataset    = PMFudanDataset(sitename=sitename, opt=opt, isTrain=False)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)
    # Model
    model = Fudan(opt)
    # Load checkpoint
    model.load_state_dict(torch.load(os.path.join(cpt_dir, f"{sitename}_{opt.model}.cpt")))
    # For device
    model.to(device)
    # Freeze model
    model.eval()
    # Parameters
    true_list  = None
    pred_list  = None
    value_list = None
    trange = tqdm(dataloader)
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, thres_y, past_data, past_ext = map(lambda z: z.to(device), data)
        # get prediction
        _, ext_pred, _ = model(x, past_data, past_ext)
        # Recover y
        ext_pred[ext_pred>= 0.5] = 1
        ext_pred[ext_pred<  0.5] = 0
        ext_pred  = ext_pred.detach().cpu().numpy()
        ext_true  = ext_true.detach().cpu().numpy()
        # Append result
        if value_list is None:
            true_list  = ext_true
            pred_list  = ext_pred
        else:
            true_list  = np.concatenate((true_list, ext_true),   axis=0)
            pred_list  = np.concatenate((pred_list, ext_pred),   axis=0)
    # Save results
    np.save(f"{rst_dir}/{sitename}.npy", pred_list)
    j = -1
    precision, recall, f1, macro, micro, weighted, mcc = get_score(true_list[:, j], pred_list[:, j])
    results.append({
        'sitename': sitename,
        'precision': f"{precision:.3f}",
        'recall'   : f"{recall   :.3f}",
        'f1'       : f"{f1       :.3f}",
        'mcc'      : f"{mcc      :.3f}",
    })
df = pd.DataFrame(results) 
df.to_csv(f"{rst_dir}/{no}_qa.csv", index=False, encoding='utf_8_sig')
print(f"Finish testing no: {no}, cost time: {datetime.now()-st_time}")
