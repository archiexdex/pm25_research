from utils import *
from constants import *
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import PMDataset
import csv, json
import pandas as pd
from dotted.collection import DottedDict

# Test
opt = parse()
with open(f"{opt.cfg_dir}/{opt.no}_{opt.method}.json", "r") as fp:
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

name_list = []
data_list = {"rmse": [], "f1": [], "micro": [], "macro": [], "weighted": []}
for sitename in sitenames:
    if opt.skip_site and sitename not in sample_sites:
        continue
    print(sitename)
    test_dataset    = PMDataset(sitename=sitename, config=opt, isTrain=False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    
    if method == "merged":
        nor_model = load_model(os.path.join(f"checkpoints/88/{sitename}_normal.pt"),  opt).to(device)
        ext_model = load_model(os.path.join(f"checkpoints/89/{sitename}_extreme.pt"), opt).to(device)
        # MARK: - learnable model or fronzen?
        ext_model.eval()
        nor_model.eval()
        
        model = DNN_merged(
            opt=opt,
            nor_model=nor_model,
            ext_model=ext_model, 
        )
        checkpoint = torch.load(os.path.join(cpt_dir, f"{sitename}_merged.pt"))
        model.load_state_dict(checkpoint)
        model.to(device)
    else:
        model = load_model(os.path.join(cpt_dir, f"{sitename}_{method}.pt"), opt).to(device)

    model.eval()
    mse = nn.MSELoss()

    st_time = datetime.now()
    mean_rmse_loss = 0
    value_list = None
    trange = tqdm(test_dataloader)
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, thres_y = map(lambda z: z.to(device), data)
        y_true  = y_true 
        thres_y = thres_y
        #y_true  = y_true [:, 0]
        #thres_y = thres_y[:, 0]
        # get loss & update
        if opt.method == "merged":
            y_pred = model(x)
        else:
            _, _, _, y_pred = model(x)
        y_pred = y_pred
        # Calculate loss
        mse_loss  = mse(y_pred * thres_y, y_true * thres_y)
        # Record loss
        mean_rmse_loss += (torch.sqrt(mse_loss)).item()
        # Recover y
        recover_y = y_pred * thres_y
        recover_y = recover_y.cpu().detach().numpy()
        # Append result
        if value_list is None:
            value_list = recover_y
        else:
            value_list = np.concatenate((value_list, recover_y), axis=0)
        trange.set_description(f"Test mean rmse: {mean_rmse_loss / (idx+1):.3f}")
    test_loss = mean_rmse_loss / len(test_dataloader)
    
    #f1, macro, micro, weighted = get_score(true_list, pred_list)
    #name_list.append(sitename)
    #data_list["rmse"].append(test_loss)
    #data_list["f1"].append(f1); data_list["macro"].append(macro)
    #data_list["micro"].append(micro); data_list["weighted"].append(weighted)
    #print(f"f1: {f1}, macro: {macro}, micro: {micro}, weighted: {weighted}")
    np.save(f"{rst_dir}/{sitename}.npy", value_list)

# save quantitative analysis
#df = pd.DataFrame({
#    "sitename": name_list,
#    "rmse":     data_list["rmse"],
#    "f1":       data_list["f1"],
#    "micro":    data_list["micro"],
#    "macro":    data_list["macro"],
#    "weighted": data_list["weighted"]
#})
#df.to_csv(f"{save_dir}/{no}_{method}.csv", index=False, encoding='utf_8_sig')
