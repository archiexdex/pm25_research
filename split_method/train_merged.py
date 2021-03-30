from utils import *
from constants import *
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import PMExtDataset, PMDataset


# Train
opt = parse()
same_seeds(opt.seed)
save_config(opt, opt.config_dir, str(opt.no))

if opt.no is not None:
    no = opt.no
else:
    print("n is not a number")
    exit()

cpt_dir = get_path(opt.cpt_dir, mode=0)
cpt_dir = get_path(cpt_dir, no, mode=0)
log_dir = get_path(opt.log_dir, mode=0)

device = get_device()

def get_model(path):
    checkpoint = torch.load(path)
    model = DNN(
        input_dim=opt.input_dim, 
        emb_dim=opt.emb_dim, 
        hid_dim=opt.hid_dim, 
        output_dim=opt.output_dim,
        source_size=opt.source_size
    )
    model.load_state_dict(checkpoint)
    return model

train_records = {}
for sitename in sitenames:
    if opt.skip_site == 1 and sitename not in sample_sites:
        continue
    print(sitename)
    
    # train combine all 
    train_dataset = PMDataset(sitename=sitename, config=opt, isTrain=True)
    valid_dataset = PMDataset(sitename=sitename, config=opt, isTrain=False)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)
    
    ext_model = get_model(os.path.join(cpt_dir, f"{sitename}_extreme.pt"))
    nor_model = get_model(os.path.join(cpt_dir, f"{sitename}_normal.pt"))
    # MARK: - learnable model or fronzen?
    ext_model.eval()
    nor_model.eval()
    
    model = DNN_merged(
        ext_model=ext_model, 
        nor_model=nor_model,
        output_dim=opt.output_dim,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    total_epoch = opt.total_epoch
    patience = opt.patience
    best_rmse = 1e9
    earlystop_counter = 0
    st_time = datetime.now()
    
    for epoch in range(total_epoch):
        train_loss = train(model, train_dataloader, mse, optimizer)
        valid_loss = test(model, valid_dataloader, mse)
        if best_rmse > valid_loss:
            best_rmse = valid_loss
            torch.save(model.state_dict(), f"checkpoint.pt")
            earlystop_counter = 0
            print(f">> Model saved epoch: {epoch}!!")

        # if best_loss doesn't improve for patience times, terminate training
        else:
            earlystop_counter += 1
            if earlystop_counter >= patience:
                print("Early stop!!!")
                print(f"sitename: {sitename}\nepoch: {epoch}\nbest_rmse: {best_rmse: .4f}")
                os.rename("checkpoint.pt", os.path.join(cpt_dir, f"{sitename}_merged.pt"))
                train_records[sitename] = {
                    "mode": "merged",
                    "best_rmse": f"{best_rmse:.3f}", 
                    "epoch": epoch, 
                    "timestamp": datetime.now() - st_time
                }
                break


# Write Record
write_record(f"{opt.log_dir}/{no}_merged.csv", train_records)

print("Done!!!")
