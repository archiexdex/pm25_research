from utils import *
from constants import *
import itertools
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import *
from model import *
import csv

opt = parse()
opt.method = 'merged_transformer'
opt.model = "transformer"
same_seeds(opt.seed)

if opt.no is not None:
    no = opt.no
else:
    print("n is not a number")
    exit()

try:
    cfg_dir = os.makedirs(os.path.join(opt.cfg_dir), 0o777)
except:
    pass
cpt_dir = os.path.join(opt.cpt_dir, str(no))
log_dir = os.path.join(opt.log_dir, str(no))
if (not opt.yes) and os.path.exists(cpt_dir):
    res = input(f"no: {no} exists, are you sure continue training? It will override all files.[y:N]")
    res = res.lower()
    if res not in ["y", "yes"]:
        print("Stop training")
        exit()
    print("Override all files.")
if os.path.exists(cpt_dir):
    shutil.rmtree(cpt_dir)
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(cpt_dir, 0o777)
os.makedirs(log_dir, 0o777)
save_config(opt)

def load_sa(opt, name, sitename):
    load_path = os.path.join(opt.cpt_dir, name, f"{sitename}_sa.cpt")
    checkpoint = torch.load(path)
    model = get_model(opt)
    model.load_state_dict(checkpoint)
    return model
    

device = get_device()
st_t = datetime.now()
train_records = {}
for sitename in SITENAMES:
    if opt.skip_site == 1 and sitename not in SAMPLE_SITES:
        continue
    print(sitename)
    
    # Dataset
    train_dataset = PMSADataset(sitename=sitename, opt=opt, isTrain=True)
    valid_dataset = PMSADataset(sitename=sitename, opt=opt, isTrain=False)

    # Get ratio
    opt.ratio = train_dataset.get_ratio()
    print(f"Extreme Event: {1-opt.ratio:.3%}, Normal Event: {opt.ratio:.3%}")

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)
    
    # Model
    assert opt.nor_load_model != None, f"Merged method should determine the load model"
    assert opt.ext_load_model != None, f"Merged method should determine the load model"
    nor_load_path = os.path.join(opt.cpt_dir, str(opt.nor_load_model), f"{sitename}_sa.cpt")
    ext_load_path = os.path.join(opt.cpt_dir, str(opt.ext_load_model), f"{sitename}_sa.cpt")
    nor_model = load_model(nor_load_path, opt, device)
    ext_model = load_model(ext_load_path, opt, device)
    for p in nor_model.parameters():
        p.requires_grad = False
    for p in ext_model.parameters():
        p.requires_grad = False
    model = Merged_Transformer(opt, nor_model, ext_model).to(device)
    # Optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    # Parameters
    total_epoch = opt.total_epoch
    patience = opt.patience
    best_loss = 1e9
    earlystop_counter = 0
    st_time = datetime.now()
    
    for epoch in range(total_epoch):
        train_loss = sa_train(opt, train_dataloader, model, optimizer, device)
        valid_loss = sa_test (opt, valid_dataloader, model, device)
        if best_loss > valid_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(cpt_dir, f"{sitename}_{opt.method}.cpt"))
            earlystop_counter = 0
            print(f">> Model saved epoch: {epoch}!!")

        # if best_loss doesn't improve for patience times, terminate training
        else:
            earlystop_counter += 1
            if patience > 0 and earlystop_counter >= patience:
                print("Early stop!!!")
                break
    print(f"sitename: {sitename}\nepoch: {epoch}\nbest_loss: {best_loss: .4f}")
print(f"Finish training no: {no}, cost time: {datetime.now() - st_t}!!!")