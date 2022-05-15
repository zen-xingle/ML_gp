from matplotlib.lines import Line2D
import xlrd
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.pyplot import MultipleLocator

def get_data(file, type):
    data = pd.read_csv(file)
    target_list = data[type]
    return target_list.values

def get_mean_and_std(method, file, data, interp, type, n):
    m = []
    s = []
    val = []
    seed = ['0', '1', '2', '3', '4']
    if method == 'dmfal':
        for i in seed:
            f = "exp/" + method + "/" + file + "/" + data + "_Seed[" + i + "]_" + interp + ".csv"
            val.append(get_data(f, 'r2'))
    else:
        for i in seed:
            f = "exp/" + method + "/" + file + "/" + data + "_Seed[" + i + "]_" + interp + ".csv"
            val.append(get_data(f, type))

    for i in range(n):
        temp = []
        for j in range(len(seed)):
            temp.append(val[j][i])
        temp = np.array(temp)
        m.append(temp.mean())
        s.append(temp.std())

    return np.array(m), np.array(s)

def makedir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

if __name__ == '__main__':
    # ratio = 0.4
    # method = ['GAR','LarGP','SGAR', 'ResGP', 'dmfal'] #对齐数据
    # method = ['GAR', 'LarGP', 'NAR', 'SGAR', 'ResGP']
    method = ['LarGP', 'ResGP', 'dmfal', 'NAR', 'DC', 'SGAR', 'GAR'] #不对齐数据
    # method = ['GAR', 'SGAR', 'NAR']
    #TopOP_mfGent_v5_Seed[0]_Interp[True]
    file_name = 'burger_v4_02_l2h_32_int'
    data_name = 'burger_v4_02'
    interp = 'Interp[True]'
    max_num = 32

    dic= {32: 4, 64:5, 128:6}

    color_dic = {'GAR':'#DC143C', 'dmfal':'#2ca02c', 'SGAR':'#1f77b4', 'LarGP':'#ff7f0e', 'ResGP':'#8c564b', 'NAR':'#708090', 'DC':'#17becf'}
    marker_dic = {'GAR':"o", 'dmfal':"s", 'SGAR':"^", 'LarGP':"v", 'ResGP':"P", 'NAR':"d", 'DC':"h"}
    ls_dic = {'GAR':'dashed', 'dmfal':'dotted', 'SGAR':'dashed', 'LarGP':'dotted', 'ResGP':'dotted', 'NAR':'dotted', 'DC':'dotted'}
    label_dic = {'GAR':'GAR', 'dmfal':'MF-BNN', 'SGAR':'CIGAR', 'LarGP':'AR', 'ResGP':'ResGP', 'NAR':'NAR', 'DC':'DC'}

    marker = ["o", "s", "^", "v", "*", "d", "h", "p", "x", "+"]
    color = ['#DC143C', '#1f77b4', '#2ca02c', '#ff7f0e', '#8c564b', '#708090', '#7f7f7f', '#000000', '#17becf']  # bcbd22
    orders = [2 ** (i + 2) for i in range(dic[max_num])]
    vals = []
    vars = []
    for i in range(len(method)):
        m, s = get_mean_and_std(method[i], file_name, data_name , interp, 'rmse', dic[max_num])
        vals.append(m)
        vars.append(s)
        plt.errorbar(orders, vals[i], yerr = vars[i], ls = ls_dic[method[i]], linewidth=3, color=color_dic[method[i]], label= label_dic[method[i]], marker=MarkerStyle(marker_dic[method[i]], fillstyle='full'), elinewidth = 2 ,capsize = 2, markersize = 10, alpha = 0.8)
        # plt.plot(orders, vals[i], linewidth=2, color=color[i], label=method[i], marker=marker[i])
        # plt.fill_between(orders, vals[i] - vars[i] * ratio, vals[i] + vars[i] * ratio, alpha=0.001, color=color[i])

    plt.xlabel("num of high-fidelity training sample", fontsize=14)
    plt.ylabel("RMSE", fontsize = 14)
    ax = plt.gca()
    plt.xticks(orders)
    plt.tick_params(axis='both', labelsize=13)
    # plt.legend(loc='upper right', fontsize=12)
    # plt.title(file_name,fontsize = 12)
    plt.grid()
    # plt.show()

    # makedir(r"fig")
    fig_file = r"fig_" + file_name + ".eps"
    plt.savefig(fig_file, bbox_inches = 'tight')


    