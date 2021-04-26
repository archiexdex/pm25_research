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

no = opt.no
method = opt.method
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
for sitename in sitenames:
    if opt.skip_site and sitename not in sample_sites:
        continue
    print(sitename)
    test_dataset    = PMDataset(sitename=sitename, config=opt, isTrain=False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    
    if method == "all":
        model = load_model(os.path.join(cpt_dir, f"{sitename}_all.pt"), model_name, opt).to(device)
    elif method == "merged":
        ext_model = load_model(os.path.join(cpt_dir, f"{sitename}_extreme.pt"), model_name, opt).to(device)
        nor_model = load_model(os.path.join(cpt_dir, f"{sitename}_normal.pt"),  model_name, opt).to(device)
        # MARK: - learnable model or fronzen?
        ext_model.eval()
        nor_model.eval()
        
        model = DNN_merged(
            ext_model=ext_model, 
            nor_model=nor_model,
            input_dim=opt.input_dim,
            output_dim=opt.output_dim,
            source_size=opt.source_size,
        )
        checkpoint = torch.load(os.path.join(cpt_dir, f"{sitename}_merged.pt"))
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
        y_true = y_true[:, -1]
        thres_y = thres_y[:, -1]
        # get loss & update
        y_pred = model(x)
        # Calculate loss
        mse_loss  = mse(y_pred * thres_y, y_true * thres_y)
        # Record loss
        mean_rmse_loss += (torch.sqrt(mse_loss)).item()
        # Recover y
        recover_y = y_pred * thres_y
        recover_y = recover_y.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().numpy()
        # For quantitative analysis
        y_pred[y_pred>=.5] = 1
        y_pred[y_pred<.5] = 0
        y_true[y_true>=1] = 1
        y_true[y_true<1] = 0
        # Append result
        if pred_list is None:
            pred_list = y_pred
            true_list = y_true
            value_list = recover_y
        else:
            pred_list  = np.concatenate((pred_list, y_pred), axis=0)
            true_list  = np.concatenate((true_list, y_true), axis=0)
            value_list = np.concatenate((value_list, recover_y), axis=0)
        trange.set_description(f"Test mean rmse: {mean_rmse_loss / (idx+1):.3f}")
    test_loss = mean_rmse_loss / len(test_dataloader)
    
    f1, macro, micro, weighted = get_score(true_list, pred_list)
    name_list.append(sitename)
    data_list["rmse"].append(test_loss)
    data_list["f1"].append(f1); data_list["macro"].append(macro)
    data_list["micro"].append(micro); data_list["weighted"].append(weighted)
#     print(f"f1: {f1}, macro: {macro}, micro: {micro}, weighted: {weighted}")
    np.save(f"{save_dir}/{sitename}.npy", value_list)

# save quantitative analysis
df = pd.DataFrame({
    "sitename": name_list,
    "rmse":     data_list["rmse"],
    "f1":       data_list["f1"],
    "micro":    data_list["micro"],
    "macro":    data_list["macro"],
    "weighted": data_list["weighted"]
})
df.to_csv(f"{save_dir}/{no}_{method}.csv", index=False, encoding='utf_8_sig')
