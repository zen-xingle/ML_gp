import xlrd
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator

def get_data(file, type):
    data = pd.read_csv(file)
    target_list = data[type]
    return target_list.values

def get_d(method, file, data, interp, type, n):
    seed = 'None'
    if method == 'dmfal':
        f = "exp/" + method + "/" + file + "/" + data + "_Seed[" + seed + "]_" + interp + ".csv"
        return get_data(f, 'r2')
    else:
        f = "exp/" + method + "/" + file + "/" + data + "_Seed[" + seed + "]_" + interp + ".csv"
        return get_data(f, type)

def makedir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

if __name__ == '__main__':
    # ratio = 0.4
    # method = ['GAR', 'LarGP','SGAR', 'ResGP', 'dmfal'] #对齐数据
    # method = ['GAR', 'LarGP', 'NAR', 'SGAR', 'ResGP']
    method = ['GAR', 'dmfal','SGAR', 'NAR'] #不对齐数据
    # method = ['GAR', 'SGAR', 'NAR']
    file_name = 'Heat_mfGent_v5_m2h_32'
    data_name = 'Heat_mfGent_v5'
    interp = 'Interp[False]'
    max_num = 32

    dic= {32: 4, 64:5, 128:6}
    color_dic = {'GAR':'#DC143C', 'dmfal':'#1f77b4', 'SGAR':'#2ca02c', 'LarGP':'#ff7f0e', 'ResGP':'#8c564b', 'NAR':'#708090'}
    marker_dic = {'GAR':"o", 'dmfal':"s", 'SGAR':"^", 'LarGP':"v", 'ResGP':"*", 'NAR':"d"}
    marker = ["o", "s", "^", "v", "*", "d", "h", "p", "x", "+"]
    color = ['#DC143C', '#1f77b4', '#2ca02c', '#ff7f0e', '#8c564b', '#708090', '#7f7f7f', '#000000', '#17becf']  # bcbd22
    orders = [2 ** (i + 2) for i in range(dic[max_num])]
    vals = []
    vars = []
    for i in range(len(method)):
        temp = get_d(method[i], file_name, data_name , interp, 'rmse', dic[max_num])
        plt.plot(orders, temp, linewidth=3, color=color_dic[method[i]], label=method[i], marker=marker_dic[method[i]], markersize = 10)
        # plt.plot(orders, vals[i], linewidth=2, color=color[i], label=method[i], marker=marker[i])
        # plt.fill_between(orders, vals[i] - vars[i] * ratio, vals[i] + vars[i] * ratio, alpha=0.001, color=color[i])

    plt.xlabel("num of high-fidelity training sample", fontsize=14)
    plt.ylabel("RMSE", fontsize = 14)
    ax = plt.gca()
    plt.tick_params(axis='both', labelsize=10)
    plt.legend(loc='upper right', fontsize=12)
    # plt.title(file_name,fontsize = 12)
    plt.grid()
    # plt.show()

    # makedir(r"fig")
    fig_file = r"fig_" + file_name + "_none" + ".eps"
    plt.savefig(fig_file, bbox_inches = 'tight')


    