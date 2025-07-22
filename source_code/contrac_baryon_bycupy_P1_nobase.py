##!/beegfs/home/liuming/software/install/python/bin/python3
import numpy as np
import cupy as cp
import math
import os
import fileinput
from gamma_cupy import *
from gamma_matrix_cupy import *
from input_output import *
from VVV_tools import *
from opt_einsum import contract
import time
import multiprocessing
import sys

# ------------------------------------------------------------------------------

tstep = 1
Nc = 3

g5g0 = gamma_matrix(5)
g5g4 = -gamma_matrix(15)
g5g5 = gamma_matrix(0)
g5g4g5 = -gamma_matrix(4)
g5gi = -gamma_matrix(12)
g5g4gi = gamma_matrix(6)
g5gig5 = -gamma_matrix(1)
g5gigj = cp.matmul(gamma_matrix(5),gamma_matrix(8))
g2=gamma_matrix(2)
g4=gamma_matrix(4)
g0=gamma_matrix(0)
g1=gamma_matrix(1)


g0g5 = g5g0
g4g5 = -g5g4
# g5g5 = g5g5
g4g5g5 = g4
gig5 = -g5gi
g4gig5 = g5g4gi
gig5g5 = -g5gig5
gigjg5 = g5gigj
C_g5=-1 *cp.matmul(g2,g4g5)
g1g3=cp.matmul(g1,g3)
P_matrix1=0.5 * (g0+g4)
P_matrix2=0.5 * (g0-g4)



#meson_list = ["g0", "g4", "g5", "g4g5", "gi", "g4gi", "gig5", "gigj"]
meson_list = ["I_0", "I_1", "I_1_1"]
#              +        -

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
    if(tmp[0] == 'nproc'):
        nproc = int(tmp[1])
    if(tmp[0] == 'Px'):    #第一个为sink动量，第二个为source动量
        Px = list(map(int, tmp[1:]))
    if(tmp[0] == 'Py'):
        Py = list(map(int, tmp[1:]))
    if(tmp[0] == 'Pz'):
        Pz = list(map(int, tmp[1:]))
    if(tmp[0] == 'peram_u_dir'):
        peram_u_dir = tmp[1]
    if(tmp[0] == 'peram_s_dir'):
        peram_s_dir = tmp[1]
    if(tmp[0] == 'peram_c_dir'):
        peram_c_dir = tmp[1]
    if(tmp[0] == 'eigen_dir'):
        eig_dir = tmp[1]
    if(tmp[0] == 'corr_dir'):
        corr_dir = tmp[1]
    if(tmp[0] == 'phi_dir'):
        phi_dir = tmp[1]

# ------------------------------------------------------------------------------



def readin_eigvecs(eig_dir, t, Nev, Nev1, conf_id, Nx):#t 时刻的eigenvector，而非直接把所有的t都求了
    f = open("%s/eigvecs_t%03d_%s" % (eig_dir, t, conf_id), 'rb')
    eigvecs = np.fromfile(f, dtype='f8')
    f.close()

    eigvecs_size = eigvecs.size
    Nev = int(eigvecs_size/(Nx*Nx*Nx*3*2))

    eigvecs = eigvecs.reshape(Nev, Nx*Nx*Nx*3, 2)
    eigvecs = eigvecs[..., 0]+eigvecs[..., 1]*1j
    eigvecs = eigvecs[0:Nev1, :]
    eigvecs = np.transpose(eigvecs)
    # eigvecs=eigvecs.reshape(Nev1,Nx*Nx*Nx*3)
    eigvecs_cupy = cp.asarray(eigvecs)
    del eigvecs
    return eigvecs_cupy


def epx(Nc, Nx, Px, Py, Pz):  #Nc是色，Nx是空间维度，Px,Py,Pz是动量？
    Mom = np.array([Pz, Py, Px])*2.0*np.pi*1j/Nx  #Mon是动量？为什么这么算呢？

    exp_diag = np.zeros(Nx*Nx*Nx*Nc, dtype=complex)
    for z in range(0, Nx):
        for y in range(0, Nx):
            for x in range(0, Nx):
                Pos = np.array([z, y, x])
                exp_diag[z*Nx*Nx*3 + y*Nx*3 + x *3] = np.exp(-np.dot(Mom, Pos))
                exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3 +1] = exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3]#色空间的所以值相同
                exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3 +2] = exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3]
    exp_diag_cupy = cp.asarray(exp_diag)
    del exp_diag
    return exp_diag_cupy


def VDV(t, exp_diags):
    eigvecs = readin_eigvecs(eig_dir, t, Nev, Nev1, conf_id, Nx)
    eigvecs_dagger = cp.conj(cp.transpose(eigvecs))
    n = exp_diags.shape[0]#exp_diags的数是动量
    vdv = cp.zeros((n, Nev1, Nev1), dtype=complex)
    for i in range(n):
        vdv[i] = cp.matmul((eigvecs_dagger * exp_diags[i]), eigvecs)
    return vdv

# ------------------------------------------------------------------------------


def readin_phi_vdv(phi_dir, t_source, Nev1, Px, Py, Pz, conf_id):#读取t时刻的phi_vdv

    f = open("%s/VVV.t%03i.Px%iPy%iPz%i.conf%s" % (phi_dir, t_source, Px, Py, Pz, conf_id), 'rb')
    phi_vdv = np.fromfile(f, dtype='f8')
    Nev=int(np.cbrt(phi_vdv.size/2))
    phi_vdv = phi_vdv.reshape(Nev, Nev, Nev, 2)
    phi_vdv = phi_vdv[..., 0] + phi_vdv[..., 1] * 1j
    phi_vdv = phi_vdv[0:Nev1, 0:Nev1, 0:Nev1]
    f.close()
    phi_vdv_cupy=cp.array(phi_vdv)
    return phi_vdv_cupy

# ------------------------------------------------------------------------------
def readin_all_phi_vdv(phi_dir, Nt, Nev1, conf_id, p_num):#读取所有t时刻的phi_vdv

    phi_vdv_all = np.zeros((p_num, Nt, Nev1, Nev1, Nev1), dtype=complex)
    for k in range(p_num):
        for t in range(Nt):
            print("reading phi_vdv, t=%d" % t)
            f = open("%s/VVV.t%03i.Px%iPy%iPz%i.conf%s" % (phi_dir, t, Px[k], Py[k], Pz[k], conf_id), 'rb')
            VVV = np.fromfile(f, dtype='f8')
            Nev=int(np.cbrt(VVV.size/2))
            VVV = VVV.reshape(Nev, Nev, Nev, 2)
            VVV = VVV[..., 0] + VVV[..., 1] * 1j
            VVV = VVV[0:Nev1, 0:Nev1, 0:Nev1]
            phi_vdv_all[k,t,:,:,:] =VVV


    phi_vdv_cupy = cp.array(phi_vdv_all)
    del phi_vdv_all
    return phi_vdv_cupy


# ------------------------------------------------------------------------------
#计算逻辑：因为为S波，因此角动量和自旋可以分开算。并且在计算角动量部分时可以运用对称性来计算，如s波本质只有+-z，x方向
#为了节约内存，在此选择使用现算的方式来计算phi_vdv，以及在每个时间片现用vvv
#但是在转动时，关联函数可能会出一个相位产生干涉，因此选择先计算c62中关联函数再组合
#三个不可约表示一共要计算
# ------------------------------------------------------------------------------


if __name__ == '__main__':

    #corr时间：20分钟，VVV时间22秒，VDV时间7秒
    meson_list = ["I0", "I1", "I1_1"]#同位旋，因为较为简单因此依旧算三个，较复杂场景会算两个
    irrep_num=2#不可约表示的数目
    #不同不可约算符的维数,决定了关联矩阵的维数
    #不采用base分析而是直接计算

#--------------------------------------------------------------------------------
    multiprocessing.set_start_method('spawn')
    print("Px: %d, Py: %d, Pz: %d" % (Px[0], Py[0], Pz[0]))
    st = time.time()
    p_num= len(Px)
    if p_num != len(Py) or p_num != len(Pz):
        print("Error: Px, Py, Pz should have the same length.")
        exit(1)
    print("p_num: %d" % p_num)

    #因为对称性，所有的关联函数可以由少数几个关联函数来表示，而为了减少计算量，p_start在此为000，001，p_end为全部的p
    #以下计算关联函数基底
    #节约空间，先计算p_start的信息,p_start的个数应该与不可约表示的数目相同,因为不使用debug节点，因此也无需分开成为不同任务

    #尝试实现计算所有的VVV，但是可能会爆内存（这里没有爆内存）因此可以提前计算
    VVV_all = cp.zeros((p_num, Nt, Nev1, Nev1, Nev1), dtype=complex)
    st = time.time()
    args = [(t, k) for t in range(0, Nt, tstep) for k in range(p_num)]
    with multiprocessing.Pool(processes=nproc) as pool:
        results = []
        for t, k in args:
            # 使用异步方式提交任务，不阻塞主进程
            result = pool.apply_async(VVV_cal, (eig_dir, Nx, t, Nev1, conf_id, Px[k], Py[k], Pz[k]))
            results.append((t, k, result))

        for t, k, result in results:
            VVV_all[k, t, ...] = result.get()

    ed = time.time()
    print("VVV_all time used: %.3f s" % (ed-st))

    exp_diags = cp.zeros((p_num, Nc*Nx**3), dtype=complex)
    for i in range(p_num):
        exp_diags[i] = epx(Nc, Nx, -Px[i], -Py[i], -Pz[i])    #动量方向相反才给了特殊的性质
    ed = time.time()
    print("exp time used: %.3f s" % (ed-st))

    VDV_all = cp.zeros((p_num, Nt, Nev1, Nev1), dtype=complex)
    st = time.time()
    args = [t for t in range(Nt)]
    with multiprocessing.Pool(processes=nproc) as pool:
        results = []
        for t in args:
            # 使用异步方式提交任务，不阻塞主进程
            result = pool.apply_async(VDV, (t, exp_diags))
            results.append(result)

        for t, result in enumerate(results):
            VDV_all[:, t, ...] = result.get()
    ed = time.time()
    print("VDV time used: %.3f s" % (ed-st))#得到所有的VDV矩阵





    # ------------------------------------------------------------------------------

    contrac_meson_temp = cp.zeros((len(meson_list) ,Nt ,Nt ,irrep_num,irrep_num), dtype=complex)
#    for t_source in range(0,Nt,tstep):
    for t_source in range(0, Nt, tstep):#t_source是源点时间
        #整理数据


        peram_u = readin_peram(peram_u_dir, conf_id, Nt, Nev1, Nev1, t_source)#读取传播子
        peram_s = readin_peram(peram_s_dir, conf_id, Nt, Nev1, Nev1, t_source)#读取传播子
        VDV_source = VDV_all[:,t_source,...]
        conj_VDV_source = cp.conj(VDV_source)
        conj_VDV_source_T = conj_VDV_source.transpose(0,2,1)
        conj_VDV_source_0 = conj_VDV_source_T[0]
        conj_VDV_source_1 = conj_VDV_source_T[1:]
        conj_VVV_source_0 = cp.conj(VVV_all[0, t_source, ...])
        conj_VVV_source_1 = cp.conj(VVV_all[1:, t_source, ...])#VVV的conj


        CG5_peram_d_CG5 =contract("ba,tzyad,ed->tzybe",  C_g5, peram_u, C_g5,)#缩并d夸克
        G5_peram_s_G5 =contract("ba,tzyda,de->tzybe", -g5g5, cp.conj(peram_s), g5g5)

        #开始计算
        for t_sink in range(0, Nt, tstep):
            #基本数据
            KnnK = cp.zeros((irrep_num,irrep_num), dtype=complex)#每次归零使用
            KnpK = cp.zeros((irrep_num,irrep_num), dtype=complex)
            peram_u_sink=peram_u[t_sink]#读取传播子
            G5_peram_s_G5_sink=G5_peram_s_G5[t_sink]#读取传播子
            CG5_peram_d_CG5_sink=CG5_peram_d_CG5[t_sink]#d夸克传播子
            VDV_sink_0=VDV_all[0,t_sink,...]
            VDV_sink_1=VDV_all[1:,t_sink,...]#VDV的conj
            VVV_sink_0 = VVV_all[0, t_sink, ...]#读取sinkde VVV
            VVV_sink_1 = VVV_all[1:,t_sink, ...]#VVV的conj
#-------------------------------------------------------------------------------------------------
            #计算关联函数
            #(0,0)
            KnnK[0,0] = -\
                contract("ijk,op,gc,ab, ilcg,jmae, prdh,knbf, oqhd, ef,rq,lmn", VVV_sink_0, VDV_sink_0, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_0, conj_VVV_source_0) -\
                contract("ijk,op,gc,ab, imce,jlag, pndf,krbh, oqhd, ef,rq,lmn", VVV_sink_0, VDV_sink_0, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_0, conj_VVV_source_0) +\
                contract("ijk,op,gc,ab, imce,jlag, prdh,knbf, oqhd, ef,rq,lmn", VVV_sink_0, VDV_sink_0, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_0, conj_VVV_source_0) +\
                contract("ijk,op,gc,ab, ilcg,jmae, pndf,krbh, oqhd, ef,rq,lmn", VVV_sink_0, VDV_sink_0, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_0, conj_VVV_source_0)
            KnpK[0,0] = +\
                contract("ijk,op,gc,ab, incf,jrah, pmde,klbg, oqhd, ef,rq,lmn", VVV_sink_0, VDV_sink_0, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_0, conj_VVV_source_0) +\
                contract("ijk,op,gc,ab, irch,jnaf, pldg,kmbe, oqhd, ef,rq,lmn", VVV_sink_0, VDV_sink_0, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_0, conj_VVV_source_0) -\
                contract("ijk,op,gc,ab, irch,jnaf, pmde,klbg, oqhd, ef,rq,lmn", VVV_sink_0, VDV_sink_0, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_0, conj_VVV_source_0) -\
                contract("ijk,op,gc,ab, incf,jrah, pldg,kmbe, oqhd, ef,rq,lmn", VVV_sink_0, VDV_sink_0, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_0, conj_VVV_source_0)
            contrac_meson_temp[2, t_sink,t_source,0,0] = \
                -contract("ijk,op,eb, ilbe,jmad,prcf, knad,oqfc, rq,lmn", VVV_sink_0, VDV_sink_0, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_0, conj_VVV_source_0)  \
                -contract("ijk,op,eb, imbd,jraf,plce, knad,oqfc, rq,lmn", VVV_sink_0, VDV_sink_0, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_0, conj_VVV_source_0)  \
                -contract("ijk,op,eb, irbf,jlae,pmcd, knad,oqfc, rq,lmn", VVV_sink_0, VDV_sink_0, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_0, conj_VVV_source_0)  \
                +contract("ijk,op,eb, imbd,jlae,prcf, knad,oqfc, rq,lmn", VVV_sink_0, VDV_sink_0, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_0, conj_VVV_source_0)  \
                +contract("ijk,op,eb, irbf,jmad,plce, knad,oqfc, rq,lmn", VVV_sink_0, VDV_sink_0, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_0, conj_VVV_source_0)  \
                +contract("ijk,op,eb, ilbe,jraf,pmcd, knad,oqfc, rq,lmn", VVV_sink_0, VDV_sink_0, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_0, conj_VVV_source_0)

            KnnK[0,1] = -\
                contract("ijk,op,gc,ab, ilcg,jmae, prdh,knbf, oqhd, ef,xrq,xlmn", VVV_sink_0, VDV_sink_0, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) -\
                contract("ijk,op,gc,ab, imce,jlag, pndf,krbh, oqhd, ef,xrq,xlmn", VVV_sink_0, VDV_sink_0, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) +\
                contract("ijk,op,gc,ab, imce,jlag, prdh,knbf, oqhd, ef,xrq,xlmn", VVV_sink_0, VDV_sink_0, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) +\
                contract("ijk,op,gc,ab, ilcg,jmae, pndf,krbh, oqhd, ef,xrq,xlmn", VVV_sink_0, VDV_sink_0, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1)
            KnpK[0,1] = +\
                contract("ijk,op,gc,ab, incf,jrah, pmde,klbg, oqhd, ef,xrq,xlmn", VVV_sink_0, VDV_sink_0, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) +\
                contract("ijk,op,gc,ab, irch,jnaf, pldg,kmbe, oqhd, ef,xrq,xlmn", VVV_sink_0, VDV_sink_0, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) -\
                contract("ijk,op,gc,ab, irch,jnaf, pmde,klbg, oqhd, ef,xrq,xlmn", VVV_sink_0, VDV_sink_0, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) -\
                contract("ijk,op,gc,ab, incf,jrah, pldg,kmbe, oqhd, ef,xrq,xlmn", VVV_sink_0, VDV_sink_0, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1)
            contrac_meson_temp[2, t_sink,t_source,0,1] = \
                -contract("ijk,op,eb, ilbe,jmad,prcf, knad,oqfc, xrq,xlmn", VVV_sink_0, VDV_sink_0, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_1, conj_VVV_source_1)  \
                -contract("ijk,op,eb, imbd,jraf,plce, knad,oqfc, xrq,xlmn", VVV_sink_0, VDV_sink_0, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_1, conj_VVV_source_1)  \
                -contract("ijk,op,eb, irbf,jlae,pmcd, knad,oqfc, xrq,xlmn", VVV_sink_0, VDV_sink_0, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_1, conj_VVV_source_1)  \
                +contract("ijk,op,eb, imbd,jlae,prcf, knad,oqfc, xrq,xlmn", VVV_sink_0, VDV_sink_0, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_1, conj_VVV_source_1)  \
                +contract("ijk,op,eb, irbf,jmad,plce, knad,oqfc, xrq,xlmn", VVV_sink_0, VDV_sink_0, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_1, conj_VVV_source_1)  \
                +contract("ijk,op,eb, ilbe,jraf,pmcd, knad,oqfc, xrq,xlmn", VVV_sink_0, VDV_sink_0, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_1, conj_VVV_source_1)

            KnnK[1,0] = -\
                contract("xijk,xop,gc,ab, ilcg,jmae, prdh,knbf, oqhd, ef,rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_0, conj_VVV_source_0) -\
                contract("xijk,xop,gc,ab, imce,jlag, pndf,krbh, oqhd, ef,rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_0, conj_VVV_source_0) +\
                contract("xijk,xop,gc,ab, imce,jlag, prdh,knbf, oqhd, ef,rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_0, conj_VVV_source_0) +\
                contract("xijk,xop,gc,ab, ilcg,jmae, pndf,krbh, oqhd, ef,rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_0, conj_VVV_source_0)
            KnpK[1,0] = +\
                contract("xijk,xop,gc,ab, incf,jrah, pmde,klbg, oqhd, ef,rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_0, conj_VVV_source_0) +\
                contract("xijk,xop,gc,ab, irch,jnaf, pldg,kmbe, oqhd, ef,rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_0, conj_VVV_source_0) -\
                contract("xijk,xop,gc,ab, irch,jnaf, pmde,klbg, oqhd, ef,rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_0, conj_VVV_source_0) -\
                contract("xijk,xop,gc,ab, incf,jrah, pldg,kmbe, oqhd, ef,rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_0, conj_VVV_source_0)
            contrac_meson_temp[2, t_sink,t_source,1,0] = \
                -contract("xijk,xop,eb, ilbe,jmad,prcf, knad,oqfc, rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_0, conj_VVV_source_0)  \
                -contract("xijk,xop,eb, imbd,jraf,plce, knad,oqfc, rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_0, conj_VVV_source_0)  \
                -contract("xijk,xop,eb, irbf,jlae,pmcd, knad,oqfc, rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_0, conj_VVV_source_0)  \
                +contract("xijk,xop,eb, imbd,jlae,prcf, knad,oqfc, rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_0, conj_VVV_source_0)  \
                +contract("xijk,xop,eb, irbf,jmad,plce, knad,oqfc, rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_0, conj_VVV_source_0)  \
                +contract("xijk,xop,eb, ilbe,jraf,pmcd, knad,oqfc, rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_0, conj_VVV_source_0)

            KnnK[1,1] = -\
                contract("xijk,xop,gc,ab, ilcg,jmae, prdh,knbf, oqhd, ef,yrq,ylmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) -\
                contract("xijk,xop,gc,ab, imce,jlag, pndf,krbh, oqhd, ef,yrq,ylmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) +\
                contract("xijk,xop,gc,ab, imce,jlag, prdh,knbf, oqhd, ef,yrq,ylmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) +\
                contract("xijk,xop,gc,ab, ilcg,jmae, pndf,krbh, oqhd, ef,yrq,ylmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1)
            KnpK[1,1] = +\
                contract("xijk,xop,gc,ab, incf,jrah, pmde,klbg, oqhd, ef,yrq,ylmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) +\
                contract("xijk,xop,gc,ab, irch,jnaf, pldg,kmbe, oqhd, ef,yrq,ylmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) -\
                contract("xijk,xop,gc,ab, irch,jnaf, pmde,klbg, oqhd, ef,yrq,ylmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) -\
                contract("xijk,xop,gc,ab, incf,jrah, pldg,kmbe, oqhd, ef,yrq,ylmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1)
            contrac_meson_temp[2, t_sink,t_source,1,1] = \
                -contract("xijk,xop,eb, ilbe,jmad,prcf, knad,oqfc, yrq,ylmn", VVV_sink_1, VDV_sink_1, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_1, conj_VVV_source_1)  \
                -contract("xijk,xop,eb, imbd,jraf,plce, knad,oqfc, yrq,ylmn", VVV_sink_1, VDV_sink_1, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_1, conj_VVV_source_1)  \
                -contract("xijk,xop,eb, irbf,jlae,pmcd, knad,oqfc, yrq,ylmn", VVV_sink_1, VDV_sink_1, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_1, conj_VVV_source_1)  \
                +contract("xijk,xop,eb, imbd,jlae,prcf, knad,oqfc, yrq,ylmn", VVV_sink_1, VDV_sink_1, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_1, conj_VVV_source_1)  \
                +contract("xijk,xop,eb, irbf,jmad,plce, knad,oqfc, yrq,ylmn", VVV_sink_1, VDV_sink_1, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_1, conj_VVV_source_1)  \
                +contract("xijk,xop,eb, ilbe,jraf,pmcd, knad,oqfc, yrq,ylmn", VVV_sink_1, VDV_sink_1, P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, conj_VDV_source_1, conj_VVV_source_1)

            contrac_meson_temp[0, t_sink,t_source,...] =  KnnK - KnpK
            contrac_meson_temp[1, t_sink,t_source,...] =  KnnK + KnpK





#        contrac_meson_temp[:,:,t_source] = corr(t_source, phi_vdv_all)


    ed = time.time()
    print("corr time used: %.3f s" % (ed-st))

# ------------------------------------------------------------------------------


    corr_meson = cp.zeros((len(meson_list), 1, Nt, irrep_num, irrep_num), dtype=complex)#真实的关联函数
    st0 = time.time()
    for t_source in range(0,Nt,tstep):
        for t_sink in range(Nt):
            if (t_sink < t_source):
                contrac_meson_temp[:,t_sink, t_source,...]= -1 * contrac_meson_temp[:, t_sink, t_source,...]
            corr_meson[:, 0, (t_sink-t_source+Nt) % Nt,...] += contrac_meson_temp[:, t_sink, t_source,...]#t_sink-t_source+Nt是为了保证t_sink-t_source是正数,即编时




    ed0 = time.time()
    print("corr done, time used: %.3f s" % (ed0-st0))
    corr_meson = cp.asnumpy(corr_meson)

    st0 = time.time()
    corr_meson_wirte=np.zeros((len(meson_list), 1, Nt),dtype=complex)
    #存数据
    for i in range (irrep_num):
        for j in range(irrep_num):
            corr_meson_wirte=corr_meson[:,:,:,i,j]
            for k in range(len(meson_list)):
                write_data_ascii(corr_meson_wirte[k], Nt, Nx, "%s/%02d_P%s_conf%s_test_%d%d.dat" %#写入的顺序有讲究没
                        (corr_dir, k, 1, conf_id,i,j))
    ed0 = time.time()
    print("write data done, time used: %.3f s" % (ed0-st0))
