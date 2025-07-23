##!/beegfs/home/liuming/software/install/python/bin/python3
import numpy as np
import cupy as cp
import math
import os
import fileinput
#from input_output import *
from opt_einsum import contract
import time
import multiprocessing
import sys
from VVV_tools import VVV_cal

# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
infile = fileinput.input()
for line in infile:
    tmp = line.split()
    if(tmp[0] == 'Nt'):
        Nt = int(tmp[1])
    if(tmp[0] == 'Nx'):
        Nx = int(tmp[1])
    if(tmp[0] == 'conf_id'):
        conf_id = tmp[1]
    if(tmp[0] == 'Nev'):
        Nev = int(tmp[1])
    if(tmp[0] == 'Nev1'):  # number of eigenvectors used in contraction
        Nev1 = int(tmp[1])
    if(tmp[0] == 'nMom'):
        nMom = int(tmp[1])
    if(tmp[0] == 'Px'):
        Px = int(tmp[1])
    if(tmp[0] == 'Py'):
        Py = int(tmp[1])
    if(tmp[0] == 'Pz'):
        Pz = int(tmp[1])
    if(tmp[0] == 'nproc'):
        nproc = int(tmp[1])
    if(tmp[0] == 'eigen_dir'):
        eig_dir = tmp[1]
    if(tmp[0] == 'VVV_save_dir'):
        VVV_dir = tmp[1]

#------------------------------------------------------------------------------

if __name__ == '__main__':

    print("Nev1 = %i" % Nev1, flush=True)
    print("Nev = %i" % Nev, flush=True)
    if not os.path.exists(VVV_dir):
        os.makedirs(VVV_dir)
    VVV = cp.zeros((Nt, Nev, Nev, Nev), dtype=complex)
    VVV= VVV_cal(eig_dir,Nx,Nt,Nev1,conf_id,Px,Py,Pz)
    VVV = cp.asnumpy(VVV)
    st1=time.time()
    for t in range(0, Nt):
        VVV_save=np.zeros((Nev,Nev,Nev,2),dtype='f8')
        VVV_save[...,0]=VVV[t].real
        VVV_save[...,1]=VVV[t].imag
        VVV_save.astype('f8').tofile("%s/VVV.t%03i.Px%iPy%iPz%i.conf%s" % (VVV_dir, t, Px, Py, Pz, conf_id))

    ed1=time.time()
    print("All jobs done , time used: %.3f s" % (ed1 - st1))