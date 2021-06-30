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

no = opt.no
method = 'unet'
model_name = opt.model

same_seeds(opt.seed)
cpt_dir = get_path(opt.cpt_dir, f"{no}", mode=0)
save_dir = get_path(opt.results_dir, mode=0)
save_dir = get_path(save_dir, f"{no}_{method}")


device = get_device()
#with open(f"{opt.config_dir}/{opt.no}.json", "r") as fp:
#    opt = json.load(fp)
#opt = DottedDict(opt)

name_list = []
data_list = {"rmse": [], "f1": [], "micro": [], "macro": [], "weighted": []}
total_results = ""
for sitename in sitenames:
    if opt.skip_site and sitename not in sample_sites:
        continue
    print(sitename)
    test_dataset    = PMUnetDataset(sitename=sitename, config=opt, isTrain=False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    
    model = UNet_1d(
        c_in=opt.input_dim, 
        c_hid=opt.hid_dim, 
    )
    checkpoint = torch.load(os.path.join(cpt_dir, f"{sitename}_{method}.pt"))
    model.load_state_dict(checkpoint)
    model.to(device)

    model.eval()
    mse = nn.MSELoss()

    st_time = datetime.now()
    mean_rmse_loss = 0
    pred_list = None
    true_list = None
    value_list = None
    trange = tqdm(test_dataloader)
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, thres_y = map(lambda z: z.to(device), data)
        y_true  = y_true[:, -1]
        thres_y = thres_y[:, -1]
        # get loss & update
        y_pred = model(x)[:, -1]
        # Get true value
        mse_loss = mse(y_pred * thres_y, y_true * thres_y)
        # Record loss
        mean_rmse_loss += (torch.sqrt(mse_loss)).item()
        # Recover y
        y_pred = (y_pred * thres_y).detach().cpu().numpy()
        
        # Append result
        if value_list is None:
            value_list = y_pred
        else:
            value_list = np.concatenate((value_list, y_pred), axis=0)
        trange.set_description(f"Test mean rmse: {mean_rmse_loss / (idx+1):.3e}")
    test_loss = mean_rmse_loss / len(test_dataloader)
    
    # f1, macro, micro, weighted = get_score(true_list, pred_list)
    # name_list.append(sitename)
    # data_list["rmse"].append(test_loss)
    # data_list["f1"].append(f1); data_list["macro"].append(macro)
    # data_list["micro"].append(micro); data_list["weighted"].append(weighted)
    np.save(f"{save_dir}/{sitename}.npy", value_list)
    total_results += f"{sitename} rmse: {test_loss: .3e} "

print(total_results)
print(f"Done no: {no}!!")
# save quantitative analysis
# df = pd.DataFrame({
#     "sitename": name_list,
#     "rmse":     data_list["rmse"],
#     "f1":       data_list["f1"],
#     "micro":    data_list["micro"],
#     "macro":    data_list["macro"],
#     "weighted": data_list["weighted"]
# })
# df.to_csv(f"{save_dir}/{no}_{method}.csv", index=False, encoding='utf_8_sig')
