import xlrd
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator

def get_fig_data_gar(file, data, num, seed, type):
    rbook = xlrd.open_workbook(file, formatting_info=True)  # 打开文件
    r_sheet = rbook.sheet_by_name(data)  # 通过名称获取
    d = []
    seed_dic = {'None': 3, 0:4, 1:5, 2:6, 3:7, 4:8, 'Mean':9, 'std':10}
    type_dic = {'mae':0, 'r2':1, 'rmse':2, 'll':3, 't':4}
    if num == 32:
        for i in range(0 + type_dic[type], 20, 5):
            d.append(r_sheet.row_values(i)[seed_dic[seed]])
    else:
        for i in range(20 + type_dic[type], 45, 5):
            d.append(r_sheet.row_values(i)[seed_dic[seed]])

    return d

def get_fig_data_nar(file, data, num, seed, type):
    rbook = xlrd.open_workbook(file, formatting_info=True)  # 打开文件
    r_sheet = rbook.sheet_by_name(data)  # 通过名称获取
    d = []
    seed_dic = {'None': 3, 0:4, 1:5, 2:6, 3:7, 4:8, 'Mean':9, 'std':10}
    type_dic = {'mae':0, 'r2':2, 'rmse':1, 'll':4, 't':3}
    if num == 32:
        for i in range(0 + type_dic[type], 20, 5):
            d.append(r_sheet.row_values(i)[seed_dic[seed]])
    else:
        for i in range(20 + type_dic[type], 45, 5):
            d.append(r_sheet.row_values(i)[seed_dic[seed]])

    return d

def get_data(file, type):
    data = pd.read_csv(file)
    target_list = data[type]
    return target_list.values

def get_mean_and_std(method, data, interp, type, n):
    m = []
    s = []
    val = []
    for i in ['0', '1', '2', '3', '4']:
        f = "exp/" + method + "/" + data + "/" + data + "_Seed[" + i + "]_" + interp + ".csv"
        val.append(get_data(f, type))

    for i in range(n):
        temp = []
        for j in range(5):
            temp.append(val[j][i])
        temp = np.array(temp)
        m.append(temp.mean())
        s.append(temp.std())

    return np.array(m), np.array(s)


if __name__ == '__main__':
    data_name = "Heat"
    max_num = 32
    vals = []
    vars = []
    typ = ["NAR", "LarGP", "ResGP"] #方法类型
    gar_m, gar_s = get_mean_and_std('GAR', 'Heat_mfGent_v5', 'Interp[True]', 'rmse', 4)
    vals.append(gar_m)
    vars.append(gar_s)
    sgar_m, sgar_s = get_mean_and_std('SGAR', 'Heat_mfGent_v5', 'Interp[True]', 'rmse', 4)
    vals.append(sgar_m)
    vars.append(sgar_s)
    for i in typ:
        if i == "GAR":
            name = data_name + i
            vals.append(np.array(get_fig_data_gar("/Users/aaaalison/Desktop/HOGP/result.xls", name, max_num, 'Mean', 'rmse')))
            vars.append(np.array(get_fig_data_gar("/Users/aaaalison/Desktop/HOGP/result.xls", name, max_num, 'std', 'rmse')))
        else:
            name = data_name + i
            vals.append(np.array(get_fig_data_nar("/Users/aaaalison/Desktop/HOGP/result.xls", name, max_num, 'Mean', 'rmse')))
            vars.append(np.array(get_fig_data_nar("/Users/aaaalison/Desktop/HOGP/result.xls", name, max_num, 'std', 'rmse')))

    result_list_32 = [4,8,16,32]
    result_list_64 = [4,8,16,32,64]
    marker = ["o", "s", "^", "v", "*", "d", "h", "p", "x", "+"]
    color = ['#DC143C', '#1f77b4', '#2ca02c', '#ff7f0e', '#8c564b', '#708090', '#7f7f7f', '#000000', '#17becf']  # bcbd22
    # val = gar + nar + lar + res + sgar
    if max_num==64:
        orders = result_list_64
    else:
        orders = result_list_32
    
    for i in range(4):
        plt.errorbar(orders, vals[i],yerr = vars[i], linewidth=2, color=color[i], label=typ[i], marker=marker[i])
        # plt.fill_between(orders, vals[i] - vars[i] * ratio, vals[i] + vars[i] * ratio, alpha=0.2, color=color[i])

    plt.xlabel("num of high-fidelity training sample", fontsize=14)
    plt.ylabel("RMSE", fontsize = 14)
    ax = plt.gca()
    plt.tick_params(axis='both', labelsize=10)
    plt.legend(loc='upper right', fontsize=12)
    plt.title("Heat_mfGent_v5",fontsize = 12)
    plt.grid()
    plt.show()
