from utils import *
from constants import *
import torch
from torch import optim
from datetime import datetime

opt = parse()
assert opt.method != None, "You must assign --method"
assert opt.model != None, "You must assign --model"
same_seeds(opt.seed)

device = get_device()
opt.device = device

check_train_id(opt)
build_dirs(opt)

save_config(opt)

st_t = datetime.now()
for sitename in SITENAMES:
    if opt.skip_site and sitename not in SAMPLE_SITES:
        continue
    print(sitename)
    # Dataset
    train_dataset = get_dataset(opt=opt, sitename=sitename, isTrain=True)
    valid_dataset = get_dataset(opt=opt, sitename=sitename, isTrain=False)
    # Get ratio
    opt.ratio = train_dataset.get_ratio()
    print(f"Extreme Event: {1-opt.ratio:.3%}, Normal Event: {opt.ratio:.3%}")
    # DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)
    # Model
    if opt.method == "merged":
        model = get_merged_model(opt, sitename).to(device)
    else:
        model = get_model(opt).to(device)
    # Loss
    loss_fn = get_loss(opt)
    # Optimizer
    if opt.method in ["merged"]:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # Parameters
    total_epoch = opt.total_epoch
    patience = opt.patience
    best_loss = 1e9
    earlystop_counter = 0
    st_time = datetime.now()
    # Trainer
    trainer = get_trainer(opt)
    
    for epoch in range(total_epoch):
        train_loss = trainer(opt, train_dataloader, model, loss_fn, optimizer)
        valid_loss = trainer(opt, valid_dataloader, model, loss_fn)
        if best_loss > valid_loss:
            best_loss = valid_loss
            torch.save( model.state_dict(), 
                        os.path.join(opt.cpt_dir, str(opt.no), f"{sitename}.cpt"))
            earlystop_counter = 0
            print(f">> Model saved epoch: {epoch}!!")

        # if best_loss doesn't improve for patience times, terminate training
        else:
            earlystop_counter += 1
            if patience > 0 and earlystop_counter >= patience:
                print("Early stop!!!")
                break
    print(f"sitename: {sitename}\nepoch: {epoch}\nbest_loss: {best_loss: .4f}")
print(f"Finish training no: {opt.no}, cost time: {datetime.now() - st_t}!!!")
