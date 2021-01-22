import numpy as np 
import seaborn as sn 
import os, shutil 
import pandas as pd 
import matplotlib.pyplot as plt
import argparse 
from constants import *

def parse():
    parser = argparse.ArgumentParser()
    try: 
        from argument import add_arguments 
        parser = add_arguments(parser)
    except:
        pass 
    return parser.parse_args()

def check_folder(path, mode="f"):
    if not os.path.exists(path):
        os.mkdir(path)
opt = parse()

if opt.no is not None:
    no = str(opt.no) 
else: 
    print("no is not a number")
    exit() 

# Check path
origin_path = opt.origin_valid_dir 
result_path = os.path.join(opt.test_results_dir, no)
save_path = opt.visual_results_dir 
if not os.path.exists(save_path):
    os.mkdir(save_path)
save_path = os.path.join(save_path, no)
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.mkdir(save_path)

plt.figure(figsize=(50, 10))
for sitename in sitenames:
    if sitename not in ["南投", "士林", "埔里", "關山"]:
        continue 
    print(f"sitename: {sitename}")
    origin_data = np.load(f"{origin_path}/{sitename}.npy")
    predict_data = np.load(f"{result_path}/{sitename}.npy")

    origin_data = origin_data[8:-8, 7:8]
    predict_data = np.expand_dims(predict_data, axis=-1)

    y = np.concatenate((origin_data, predict_data), axis=-1)

    # TODO: the date should be customized
    dates = pd.date_range("1 1 2018", periods=predict_data.shape[0], freq="H")
    data = pd.DataFrame(y, dates, columns=["origin", "predict"])
    #sn.set_theme(style="whitegrid")
    plot = sn.lineplot(data=data, palette="tab10", linewidth=2, )
    #plot.set_title(f"{sitename}") 
    plot.figure.savefig(f"{save_path}/{sitename}.png")
    #break
    plt.clf()
