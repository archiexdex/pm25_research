import pandas as pd
import numpy as np
import os, shutil
import json

"""
In the preprocess, it will read origin csv files, parse data and then save
np files what aimed to access data quickly.
"""

# MARK: - Variables

all_sitenames_76 = [
    '三義', '三重', '中壢', '中山', '二林', '仁武', '冬山', '前金', '前鎮', '南投',
    '古亭', '善化', '嘉義', '土城', '埔里', '基隆', '士林', '大同', '大園', '大寮',
    '大里', '安南', '宜蘭', '小港', '屏東', '崙背', '左營', '平鎮', '彰化', '復興',
    '忠明', '恆春', '斗六', '新店', '新港', '新營', '新竹', '新莊', '朴子', '松山',
    '板橋', '林口', '林園', '桃園', '楠梓', '橋頭', '永和', '汐止', '沙鹿', '淡水',
    '湖口', '潮州', '竹山', '竹東', '線西', '美濃', '臺南', '臺東', '臺西', '花蓮',
    '苗栗', '菜寮', '萬華', '萬里', '西屯', '觀音', '豐原', '金門', '關山', '陽明',
    '頭份', '馬公', '馬祖', '鳳山', '麥寮', '龍潭'
    ]

# size of 73
sitenames = [
    '三義', '三重', '中壢', '中山', '二林', '仁武', '冬山', '前金', '前鎮', '南投', 
    '古亭', '善化', '嘉義', '土城', '埔里', '基隆', '士林', '大同', '大園', '大寮', 
    '大里', '安南', '宜蘭', '小港', '屏東', '崙背', '左營', '平鎮', '彰化', '復興', 
    '忠明', '恆春', '斗六', '新店', '新港', '新營', '新竹', '新莊', '朴子', '松山', 
    '板橋', '林口', '林園', '桃園', '楠梓', '橋頭', '永和', '汐止', '沙鹿', '淡水', 
    '湖口', '潮州', '竹山', '竹東', '線西', '美濃', '臺南', '臺東', '臺西', '花蓮', 
    '苗栗', '菜寮', '萬華', '萬里', '西屯', '觀音', '豐原', '關山', '陽明', '頭份', 
    '鳳山', '麥寮', '龍潭'
    ]
sitenames_sorted = sorted(sitenames)

# pollutant features = ['SO2', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5']
# weather features = ['RAINFALL', 'RH', 'AMB_TEMP', 'WIND_cos', 'WIND_sin',]
feature_cols = ['SO2', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5',
                'RAINFALL', 'RH', 'AMB_TEMP', 'WIND_cos', 'WIND_sin',
                # 'sn', 'read_time'
                ]


dataset_files = ['epa_tw_14_direction.csv',
                'epa_tw_15_direction.csv',
                'epa_tw_16_direction.csv',
                'epa_tw_17_direction.csv',
                'epa_tw_18_direction.csv']

# MARK: - Functions
def read_csv(dataset):
    data = pd.DataFrame()
    if dataset == "train":
        for d in dataset_files[:4]:
            read_path = os.path.join(os.path.dirname(__file__), d)
            data = data.append(pd.read_csv(read_path, index_col='Unnamed: 0'))
    elif dataset == "valid":
        for d in dataset_files[4:]:
            read_path = os.path.join(os.path.dirname(__file__), d)
            data = data.append(pd.read_csv(read_path, index_col='Unnamed: 0'))
    elif dataset == "all":
        for d in dataset_files:
            read_path = os.path.join(os.path.dirname(__file__), d)
            data = data.append(pd.read_csv(read_path, index_col='Unnamed: 0'))
    return data

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
    data_time  = np.zeros((data.shape[0], 1))
    data_month = np.zeros((data.shape[0], 1))
    for i, d in enumerate(data['read_time'].apply(lambda x: x.day)):
        data_day[i,] = d
    for i, d in enumerate(data['read_time'].apply(lambda x: x.hour)):
        data_time[i] = d
    for i, d in enumerate(data['read_time'].apply(lambda x: x.month)):
        data_month[i] = d-1
    # Append time and extreme event buf into data_features
    data_features = np.concatenate((data_features, data_month, data_day, data_time), axis=-1)
    # Fetch site and hash
    sn_hash = dict(zip(sitenames_sorted, range(len(sitenames_sorted))))
    data_sn = np.zeros((data.shape[0], 1))
    for i, d in enumerate(data['sn']):
        data_sn[i] = sn_hash[d]
    # Split data by sitename
    data_dict = sn_hash.copy()
    data_features = data_features.reshape([-1, 73, 16])
    for i, key in enumerate(data_dict):
        data_dict[key] = data_features[:, i, :]
    return data_dict

def get_normalize(data):
    """
        Input: 
            data: dict, data[key] = [time, feature]
        Output:
            data: dict, data[key] = [time, feature]
            mean_dict: dict, mean_dict[key] = [mean]
            std_dict: dict, std_dict[key] = [std]
    """
    mean_dict = {}
    std_dict = {}
    for i, key in enumerate(data):
        mean = data[key].mean(axis=0)
        std  = data[key].std(axis=0)
        mean_dict[key] = mean.tolist()
        std_dict[key] = std.tolist()
        data[key] = (data[key] - mean) / std 
    return data, mean_dict, std_dict

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
    print("read data")
    train_data = read_csv("train")
    valid_data = read_csv("valid")

    # Filter data
    print("filter feature")
    train_data_dict = filter_data(train_data)
    valid_data_dict = filter_data(valid_data)

    # Normalize train_data_feature
    print("normalize feature")
    train_norm_data, train_mean, train_std = get_normalize(train_data_dict.copy())
    # Normalize valid_data by train
    valid_norm_data = put_normalize(valid_data_dict.copy(), train_mean, train_std)

    # Save file
    print("Save file")
    with open("train_mean.json", "w") as fp:
        json.dump(train_mean, fp, ensure_ascii=False, indent=4)
    with open("train_std.json", "w") as fp:
        json.dump(train_std,  fp, ensure_ascii=False, indent=4)
    
    # check whether the folder exists
    try:
        os.mkdir("origin")
    except:
        shutil.rmtree("origin")
        os.mkdir("origin")
    os.mkdir("origin/train")
    os.mkdir("origin/valid")
    
    try:
        os.mkdir("norm")
    except:
        shutil.rmtree("norm")
        os.mkdir("norm")
    os.mkdir("norm/train")
    os.mkdir("norm/valid")

    for key in train_data_dict:
        np.save(f"origin/train/{key}.npy", train_data_dict[key])
        np.save(f"origin/valid/{key}.npy", valid_data_dict[key])
        np.save(f"norm/train/{key}.npy", train_norm_data[key])
        np.save(f"norm/valid/{key}.npy", valid_norm_data[key])
    
