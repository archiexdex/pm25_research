import matplotlib.pyplot as plt
import os, shutil
import numpy as np
import scipy.stats as stats


read_path = "origin/train"
save_path = "analysis"
if not os.path.exists(save_path):
    os.mkdir(save_path)

for sitename in os.listdir(read_path):
    filename = os.path.join(read_path, sitename)
    sitename = sitename.split(".npy")[0]
    print(sitename)
    data = np.load(filename).astype(np.int)[:, 7]
    mean = np.mean(data)
    std = np.std(data)
    n, bins, patches = plt.hist(
        x=data, 
        #bins='auto', 
        bins=200,
        color='#0504aa',
        alpha=0.7, 
        rwidth=0.85,
        range=[0, 150],
    )
    maxfreq = n.max()
    x = np.linspace(max(mean - 4*std, 0), mean + 4*std, 200)
    y = stats.norm.pdf(x, mean, std) * maxfreq * 25 
    pos = np.where(y==max(y))
    dif = n[pos] / y[pos] 
    y *= dif
    print(y)
    print(n[pos])
    print(dif)

    plt.plot(x, y)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f' PM25 Histogram')
    plt.text(100, 800, f'mean={mean:.3f}\nstd={std:.3f}')
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig(os.path.join(save_path, sitename))
    plt.clf()
    break
