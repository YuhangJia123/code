import numpy as np
from os import listdir
from re import match
import cProfile
from functools import reduce
from check_corr import get_conf_id

iog_path = "../corr_nobase/"
T = 72

meson_list = ["I0", "I1"]
irrp_list = ["P[000]","P[001]","P[011]","P[111]","P[002]"]
chdim = len(irrp_list)
files = listdir(iog_path)
for k in range(len(meson_list)):
    iogs = []
    confg_ids_per_ij = []

    # Collect files for each ij combination and their configuration IDs
    for i in range(chdim):
        for j in range(chdim):
            iog = [file for file in files if match(r"%02d.*_%d%d.dat"%(k,i,j),file)!=None]
            confg_ids = [get_conf_id(f) for f in iog]
            confg_ids_per_ij.append(set(confg_ids))
            iog.sort()
            iogs.append(iog)
    # Find common configurations across all ij combinations
    common_confs = set.intersection(*confg_ids_per_ij) if confg_ids_per_ij else set()

    # Print incomplete configurations
    all_confs = set.union(*confg_ids_per_ij) if confg_ids_per_ij else set()
    incomplete_confs = all_confs - common_confs
    if incomplete_confs:
        print(f"不完整的组态号: {', '.join(sorted(incomplete_confs))}")

    # Filter out incomplete configurations
    filtered_iogs = []
    for idx, iog in enumerate(iogs):
        filtered_iog = [f for f in iog if get_conf_id(f) in common_confs]
        filtered_iog.sort()
        filtered_iogs.append(filtered_iog)

    iogs = filtered_iogs
    Ncnfg = len(iogs[0])
    print(Ncnfg)

    c2pt = np.zeros((chdim,chdim,Ncnfg,T),dtype=complex)
    for i in range(chdim):
        for j in range(chdim):
            for indx in range(Ncnfg):
                c2pt[i,j,indx] = np.loadtxt(iog_path+iogs[i*chdim+j][indx],skiprows=1)[:,1]
            np.save("./data/%02d_%d%d.npy"%(k,i,j),c2pt[i,j])

#    for i in range(1,chdim):
#        for indx in range(Ncnfg):
#            c2pt[0,i,indx] = np.loadtxt(iog_path+iogs[i][indx],skiprows=1)[:,2]*1j
#            c2pt[i,0,indx] = np.loadtxt(iog_path+iogs[i*chdim][indx],skiprows=1)[:,2]*1j
#        np.save("%02d_%d%d.npy"%(k,0,i),c2pt[0,i])
#        np.save("%02d_%d%d.npy"%(k,i,0),c2pt[i,0])
