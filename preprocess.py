from constants import *
from utils import *
import pandas as pd
import numpy as np
import os, shutil
import json

"""
In the preprocess, it will read origin csv files, parse data and then save
np files what aimed to access data quickly.
"""

# MARK: - Variables

opt = parse()
# size of 73
sitenames_sorted = sorted(sitenames)

# pollutant features = ['SO2', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5']
# weather features = ['RAINFALL', 'RH', 'AMB_TEMP', 'WIND_cos', 'WIND_sin',]
feature_cols = ['SO2', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5',
                'RAINFALL', 'RH', 'AMB_TEMP', 'WIND_cos', 'WIND_sin',
               ]

dataset_files = ['epa_tw_14_direction.csv',
                 'epa_tw_15_direction.csv',
                 'epa_tw_16_direction.csv',
                 'epa_tw_17_direction.csv',
                 'epa_tw_18_direction.csv',
                 'epa_tw_19_direction.csv'
                ]

# MARK: - Functions
def read_csv(dataset):
    data = pd.DataFrame()
    if dataset == "train":
        st, ed = 0, -1
    elif dataset == "valid":
        st, ed = len(dataset_files)-1, len(dataset_files)
    elif dataset == "all":
        st, ed = 0, len(dataset_files)
    data_dict = None
    for d in dataset_files[st:ed]:
        read_path = os.path.join("data", d)
        data = pd.read_csv(read_path, index_col='Unnamed: 0')
        data = filter_data(data)
        if not data_dict:
            data_dict = data
        else:
            for key in data_dict:
                data_dict[key] = np.concatenate((data_dict[key], data[key])) 
    return data_dict

def filter_data(data):
    # parse real time
    data['read_time'] = pd.to_datetime(data['read_time'])
    # Check whether the site in the sitenames_sorted list 
    data = data[data.sn.isin(sitenames_sorted)]
    # Sort data by read_time and sitename 
    data = data.sort_values(['read_time', 'sn'])
    # Reorder the data 
    data = data.reset_index(drop=True)
    # Fetch the features from data 
    data_features = data[feature_cols].values
    # Fetch the time feature
    data_day   = np.zeros((data.shape[0], 1))
    data_hour  = np.zeros((data.shape[0], 1))
    data_month = np.zeros((data.shape[0], 1))
    for i, d in enumerate(data['read_time'].apply(lambda x: x.day)):
        data_day[i,] = d
    for i, d in enumerate(data['read_time'].apply(lambda x: x.hour)):
        data_hour[i] = d
    for i, d in enumerate(data['read_time'].apply(lambda x: x.month)):
        data_month[i] = d
    # Append time and extreme event buf into data_features
    data_features = np.concatenate((data_features, data_month, data_day, data_hour), axis=-1)
    # Fetch site and hash
    sn_hash = dict(zip(sitenames_sorted, range(len(sitenames_sorted))))
    data_sn = np.zeros((data.shape[0], 1))
    for i, d in enumerate(data['sn']):
        data_sn[i] = sn_hash[d]
    # Split data by sitename
    data_dict = sn_hash.copy()
    data_features = data_features.reshape([-1, len(sn_hash), 16])
    for i, key in enumerate(data_dict):
        data_dict[key] = data_features[:, i, :]
    return data_dict

def get_normalize(data):
    """
        Input: 
            data: dict, data[key] = [time, feature]
        Output:
            mean_dict: dict, mean_dict[key] = [mean]
            std_dict: dict, std_dict[key] = [std]
            threshold_dict: dict, {key: {winter: value, summer: value}}
    """
    mean_dict = {}
    std_dict = {}
    threshold_dict = {}
    ratio = opt.ratio
    for i, key in enumerate(data):
        _data = data[key]
        # summer
        s_index = np.isin(_data[:, -3], [4,5,6,7,8,9])
        s_mean  = _data[s_index, 7].mean()
        s_std   = _data[s_index, 7].std()
        s_threshold = s_mean + s_std * ratio
        # winter
        w_index = np.isin(_data[:, -3], [4,5,6,7,8,9], invert=True)
        w_mean  = _data[w_index, 7].mean()
        w_std   = _data[w_index, 7].std()
        w_threshold = w_mean + w_std * ratio
        # global
        mean = _data.mean(axis=0)
        std  = _data.std(axis=0)
        mean_dict[key] = mean.tolist()
        std_dict[key] = std.tolist()
        threshold_dict[key] = {"winter": w_threshold, "summer": s_threshold}
        #data[key] = (data[key] - mean) / std 
    return mean_dict, std_dict, threshold_dict 

def put_normalize(data, mean_dict, std_dict):
    for i, key in enumerate(data):
        mean = np.array(mean_dict[key])
        std = np.array(std_dict[key])
        data[key] = (data[key] - mean) / std
    return data

def pm25_to_AQI(x):
    if x<15.5:
        AQI = 0
    elif x<35.5:
        AQI = 1
    elif x<54.5:
        AQI = 2
    elif x<150.5:
        AQI = 3
    elif x<250.5:
        AQI = 4
    elif x<350.5:
        AQI = 5
    else:
        AQI = 6
    return AQI

# MARK: - Main
if __name__ == '__main__':
    # Read data
    print("read data & filter feature")
    all_data = read_csv("all")
    train_data = read_csv("train")
    valid_data = read_csv("valid")

    # Normalize train_data_feature
    print("normalize feature")
    train_mean, train_std, train_threshold = get_normalize(train_data.copy())
    
    # Concat train tail and valid, and all data
    for key in train_data:
        valid_data[key] = np.concatenate((train_data[key][-opt.memory_size:], valid_data[key]), axis=0)

    # Save file
    print("Save file")
    with open("data/train_mean.json", "w") as fp:
        json.dump(train_mean, fp, ensure_ascii=False, indent=4)
    with open("data/train_std.json", "w") as fp:
        json.dump(train_std,  fp, ensure_ascii=False, indent=4)
    with open("data/train_threshold.json", "w") as fp:
        json.dump(train_threshold,  fp, ensure_ascii=False, indent=4)
    
    # check whether the folder exists
    try:
        os.mkdir("data/origin")
    except:
        shutil.rmtree("data/origin")
        os.mkdir("data/origin")
    os.mkdir("data/origin/all")
    os.mkdir("data/origin/train")
    os.mkdir("data/origin/valid")
    
    for key in train_data:
        np.save(f"data/origin/all/{key}.npy",   all_data[key])
        np.save(f"data/origin/train/{key}.npy", train_data[key])
        np.save(f"data/origin/valid/{key}.npy", valid_data[key])
    
