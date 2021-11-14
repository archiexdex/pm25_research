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

# Test
opt = parse()
with open(f"{opt.cfg_dir}/{opt.no}.json", "r") as fp:
    opt = json.load(fp)
opt = Namespace(**opt)
same_seeds(opt.seed)

no = opt.no

same_seeds(opt.seed)
cpt_dir = os.path.join(opt.cpt_dir, str(no))
rst_dir = os.path.join(opt.rst_dir, str(no))
if not os.path.exists(rst_dir):
    os.makedirs(rst_dir, 0o777)

results = []
mean_precision, mean_recall, mean_f1, mean_mcc = 0,0,0,0
for sitename in SITENAMES:
    if opt.skip_site and sitename not in SAMPLE_SITES:
        continue
    print(sitename)
    # Dataset
    dataset = get_dataset(opt=opt, sitename=sitename, isTrain=False)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)
    # Model
    model = get_model(opt)
    # Load checkpoint
    model.load_state_dict(torch.load(os.path.join(cpt_dir, f"{sitename}_{opt.method}_{opt.model}.cpt")))
    # For device
    model.to(opt.device)
    # Freeze model
    model.eval()
    # Parameters
    st_time = datetime.now()
    pred_list = None
    true_list = None
    trange = tqdm(dataloader)
    for idx, data in enumerate(trange):
        # get data
        if opt.method == "fudan":
            x, y_true, ext_true, thres_y, past_data, past_ext = map(lambda z: z.to(opt.device), data)
        else:
            x, y_true, ext_true, past_data = map(lambda z: z.to(opt.device), data)
        # get prediction
        if opt.model == "fudan":
            _, ext_pred, _ = model(x, past_data, past_ext)
        elif opt.model == "transformer":
            ext_pred, _, _ = model(past_data, x)
        elif opt.model == "seq":
            ext_pred = model(past_data, x)
        else:
            _, _, ext_pred = model(past_data, x)
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
    # Save results
    np.save(f"{rst_dir}/{sitename}.npy", pred_list)
    j = -1
    precision, recall, f1, macro, micro, weighted, mcc = get_score(true_list[:, j], pred_list[:, j])
    mean_precision += precision; mean_recall += recall; mean_f1 += f1; mean_mcc += mcc;
    results.append({
        'sitename': sitename,
        'precision': f"{precision:.4f}",
        'recall'   : f"{recall   :.4f}",
        'f1'       : f"{f1       :.4f}",
        'mcc'      : f"{mcc      :.4f}",
    })
results.insert(0, {
    'sitename': 'average',
    'precision': f"{mean_precision /len(SITENAMES):.4f}",
    'recall'   : f"{mean_recall    /len(SITENAMES):.4f}",
    'f1'       : f"{mean_f1        /len(SITENAMES):.4f}",
    'mcc'      : f"{mean_mcc       /len(SITENAMES):.4f}",
})
df = pd.DataFrame(results) 
df.to_csv(f"{rst_dir}/{no}.csv", index=False, encoding='utf_8_sig')
print(f"Finish testing no: {no}, cost time: {datetime.now()-st_time}")
