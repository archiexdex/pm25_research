import matplotlib.pyplot as plt
import os, shutil
import numpy as np
import csv
import pandas as pd
from sklearn.metrics import f1_score, precision_score

train_path = "dataset/origin/train"
valid_path = "dataset/origin/valid"
ratio = 1.5

name_list = []
data1_list = {"f1": [], "micro": [], "macro": [], "weighted": []}
data2_list = {"f1": [], "micro": [], "macro": [], "weighted": []}
data4_list = {"f1": [], "micro": [], "macro": [], "weighted": []}
data8_list = {"f1": [], "micro": [], "macro": [], "weighted": []}


for sitename in os.listdir(valid_path):
    filename = os.path.join(train_path, sitename)
    train_data = np.load(filename)
    filename = os.path.join(valid_path, sitename)
    valid_data = np.load(filename)
    sitename = sitename.split(".npy")[0]
    print(sitename)
    # summer
    s_index = np.isin(train_data[:, -3], [4,5,6,7,8,9])
    s_mean = train_data[s_index, 7].mean()
    s_std = train_data[s_index, 7].std()
    s_threshold = s_mean + s_std * ratio
    ## put to valid
    s_index = np.isin(valid_data[:, -3], [4,5,6,7,8,9])
    s_data = valid_data[s_index, 7]
    s_data[s_data <= s_threshold] = 0
    s_data[s_data > s_threshold] = 1
    valid_data[s_index, 7] = s_data
    # winter
    w_index = np.isin(train_data[:, -3], [4,5,6,7,8,9], invert=True)
    w_mean = train_data[w_index, 7].mean()
    w_std = train_data[w_index, 7].std()
    w_threshold = w_mean + w_std * ratio
    ## put to valid 
    w_index = np.isin(valid_data[:, -3], [4,5,6,7,8,9], invert=True)
    w_data = valid_data[w_index, 7]
    w_data[w_data <= w_threshold] = 0
    w_data[w_data > w_threshold] = 1
    valid_data[w_index, 7] = w_data
    # different baseline 
    data = valid_data
    origin_data  = data[8:  , 7]
    data_shift_1 = data[7:-1, 7]
    data_shift_2 = data[6:-2, 7]
    data_shift_4 = data[4:-4, 7]
    data_shift_8 = data[:-8 , 7]
    def get_score(y_true, y_pred):
        f1       = f1_score(y_true, y_pred)
        macro    = f1_score(y_true, y_pred, average='macro')
        micro    = f1_score(y_true, y_pred, average='micro')
        weighted = f1_score(y_true, y_pred, average='weighted')
        return f1, macro, micro, weighted
    f1, macro, micro, weighted = get_score(origin_data, data_shift_1)
    data1_list["f1"].append(f1); data1_list["macro"].append(macro); data1_list["micro"].append(micro); data1_list["weighted"].append(weighted)
    f1, macro, micro, weighted = get_score(origin_data, data_shift_2)
    data2_list["f1"].append(f1); data2_list["macro"].append(macro); data2_list["micro"].append(micro); data2_list["weighted"].append(weighted)
    f1, macro, micro, weighted = get_score(origin_data, data_shift_4)
    data4_list["f1"].append(f1); data4_list["macro"].append(macro); data4_list["micro"].append(micro); data4_list["weighted"].append(weighted)
    f1, macro, micro, weighted = get_score(origin_data, data_shift_8)
    data8_list["f1"].append(f1); data8_list["macro"].append(macro); data8_list["micro"].append(micro); data8_list["weighted"].append(weighted)
    name_list.append(sitename)

# write csv
df = pd.DataFrame({
    "sitename": name_list,
    "shift_1_f1":       data1_list["f1"],
    "shift_1_micro":    data1_list["micro"],
    "shift_1_macro":    data1_list["macro"],
    "shift_1_weighted": data1_list["weighted"],

    "shift_2_f1":       data2_list["f1"],
    "shift_2_micro":    data2_list["micro"],
    "shift_2_macro":    data2_list["macro"],
    "shift_2_weighted": data2_list["weighted"],

    "shift_4_f1":       data4_list["f1"],
    "shift_4_micro":    data4_list["micro"],
    "shift_4_macro":    data4_list["macro"],
    "shift_4_weighted": data4_list["weighted"],

    "shift_8_f1":       data8_list["f1"],
    "shift_8_micro":    data8_list["micro"],
    "shift_8_macro":    data8_list["macro"],
    "shift_8_weighted": data8_list["weighted"],

})
df.to_csv("base.csv", index=False)
