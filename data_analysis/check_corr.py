import numpy as np
import pandas as pd
from os import listdir
import re  # 添加正则模块用于组态ID提取
from re import match
import cProfile
from functools import reduce

def get_conf_id(filename):
    """从文件名提取组态ID（如'conf4050'）"""
    match = re.search(r'conf\d+', filename)
    return match.group(0) if match else None

iog_path = "../corr_nobase/"
T = 72

meson_list = ["I0", "I1"]
irrp_list = ["P[000]","P[001]","P[011]","P[111]","P[002]"]
chdim = len(irrp_list)
files = listdir(iog_path)
# 打开输出文件（只清空一次）
with open("check_output", "w") as output_file:
    for k in range(len(meson_list)):
        iogs = []
        conf_ids_per_ij = []

        # 收集每个ij组合的文件列表和组态ID
        for i in range(chdim):
            for j in range(chdim):
                iog = [file for file in files if match(r"%02d.*_%d%d.dat"%(k,i,j),file)!=None]
                conf_ids = [get_conf_id(f) for f in iog]
                conf_ids_per_ij.append(set(conf_ids))
                iog.sort()
                iogs.append(iog)
        # 找出在所有ij组合中都存在的完整组态
        common_confs = set.intersection(*conf_ids_per_ij) if conf_ids_per_ij else set()

        # 打印不完整的组态号（保留控制台输出）
        all_confs = set.union(*conf_ids_per_ij) if conf_ids_per_ij else set()
        incomplete_confs = all_confs - common_confs
        if incomplete_confs:
            print(f"不完整的组态号: {', '.join(sorted(incomplete_confs))}")

        # 过滤掉不完整的组态
        filtered_iogs = []
        for idx, iog in enumerate(iogs):
            filtered_iog = [f for f in iog if get_conf_id(f) in common_confs]
            filtered_iog.sort()
            filtered_iogs.append(filtered_iog)

        iogs = filtered_iogs
        Ncnfg = len(iogs[0])

        c2pt = np.zeros((chdim,chdim,Ncnfg,T),dtype=complex)
        c2pt_sum = np.zeros((chdim,chdim,T),dtype=complex)
        for i in range(chdim):
            for j in range(chdim):
                for indx in range(Ncnfg):
                    foo = np.loadtxt(iog_path+iogs[i*chdim+j][indx],skiprows=1)
                    c2pt[i,j,indx] = foo[:,1]+foo[:,2]*1j
                    c2pt_sum[i,j] = np.sum(c2pt[i,j],axis=0)/Ncnfg

        c2pt_sum_n = np.zeros((chdim,chdim,T),dtype=complex)
        for i in range(chdim):
            for j in range(chdim):
                c2pt_sum_n[i,j,1] = c2pt_sum[i,j,1] / np.sqrt(c2pt_sum[i,i,1]*c2pt_sum[j,j,1])

        # 控制台输出保持不变
        print('condiction of %d'%k)
        #print(c2pt_sum_n[:,:,1])
        print('\n')
        print(pd.DataFrame(c2pt_sum_n[:,:,1],index=irrp_list,columns=irrp_list))
        print('\n')
        
        # 写入输出文件（保持矩阵格式）
        output_file.write('condiction of %d\n'%k)
        #output_file.write(str(c2pt_sum_n[:,:,1]) + '\n\n')
        output_file.write(str(pd.DataFrame(c2pt_sum_n[:,:,1],index=irrp_list,columns=irrp_list)) + '\n\n')