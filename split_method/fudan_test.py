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
with open(f"{opt.cfg_dir}/{opt.no}_fudan.json", "r") as fp:
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
for sitename in SITENAMES:
    if opt.skip_site and sitename not in SAMPLE_SITES:
        continue
    print(sitename)
    test_dataset    = PMFudanDataset(sitename=sitename, config=opt, isTrain=False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    
    # Model
    encoder = Fudan_Encoder(opt, device)
    history = Fudan_History(opt, device)
    decoder = Fudan_Decoder(opt, device)
    # Load checkpoint
    encoder.load_state_dict(torch.load(os.path.join(cpt_dir, f"{sitename}_{method}_encoder.cpt")))
    history.load_state_dict(torch.load(os.path.join(cpt_dir, f"{sitename}_{method}_history.cpt")))
    decoder.load_state_dict(torch.load(os.path.join(cpt_dir, f"{sitename}_{method}_decoder.cpt")))
    # For device
    encoder.to(device)
    history.to(device)
    decoder.to(device)
    # Freeze model
    encoder.eval()
    history.eval()
    decoder.eval()
    # Parameters
    mseLoss = nn.MSELoss()
    st_time = datetime.now()
    mean_rmse_loss = 0
    value_list = None
    trange = tqdm(test_dataloader)
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
            # Store to buffer
            ext_preds[:, j] = ext_pred[:, 0]
            outputs[:, j]   = output[:, 0]
        # Record loss
        rmse_loss  = torch.sqrt(mseLoss(outputs * thres_ys, ys * thres_ys)) 
        mean_rmse_loss += rmse_loss.item()
        # Recover y
        recover_y = outputs * thres_ys
        recover_y = recover_y.cpu().detach().numpy()
        recover_y = recover_y[:, -1]
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
