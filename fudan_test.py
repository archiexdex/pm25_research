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

results = []
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
            output[output<0] = 0
            ext_preds[:, j] = ext_pred[:, 0]
            outputs  [:, j] = output[:, 0]
        # Record loss
        rmse_loss = torch.sqrt(mseLoss(outputs * thres_ys, ys * thres_ys)) 
        pred_loss = loss_fn(ext_preds, exts)
        mean_rmse_loss += rmse_loss.item()
        mean_pred_loss  += pred_loss.item()
        # Recover y
        recover_y = outputs * thres_ys
        ext_preds[ext_preds>= 0.05] = 1
        ext_preds[ext_preds<  0.05] = 0
        recover_y = recover_y[:, ].detach().cpu().numpy()
        pred_ext  = ext_preds[:, ].detach().cpu().numpy()
        exts      = exts     [:, ].detach().cpu().numpy()
        # Append result
        if value_list is None:
            value_list = recover_y
            true_list  = exts
            pred_list  = pred_ext
        else:
            value_list = np.concatenate((value_list, recover_y), axis=0)
            true_list  = np.concatenate((true_list, exts),       axis=0)
            pred_list  = np.concatenate((pred_list, pred_ext),   axis=0)
        trange.set_description(f"Test mean rmse: {mean_rmse_loss / (idx+1):.3f} pred: {mean_pred_loss / (idx+1):.3f}")
    #test_loss = mean_rmse_loss / len(test_dataloader)
    
    # Save the prediction value
    np.save(f"{rst_dir}/{sitename}.npy", value_list)
    np.save(f"{rst_dir}/{sitename}_class.npy", pred_list)
    # Record quantitative index
    #precision, recall, f1, macro, micro, weighted = get_score(true_list, pred_list)
    #print(f"precision: {precision}, recall: {recall}, f1: {f1}, macro: {macro}, micro: {micro}, weighted: {weighted}")
    for j in range(true_list.shape[1]):
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
