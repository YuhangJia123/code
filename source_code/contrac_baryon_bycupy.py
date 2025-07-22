##!/beegfs/home/liuming/software/install/python/bin/python3
import numpy as np
import cupy as cp
import math
import os
import fileinput
from gamma_cupy import *
from gamma_matrix_cupy import *
from input_output import *
from opt_einsum import contract
import time
import multiprocessing
from VVV_tools import *

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
S_to_DR=cp.zeros((4,4),dtype=complex)
S_to_DR[0,3]=1.0+0.0*1j
S_to_DR[1,2]=1.0+0.0*1j
S_to_DR[2,1]=1.0+0.0*1j
S_to_DR[3,0]=1.0+0.0*1j


g0g5 = g5g0
g4g5 = -g5g4
# g5g5 = g5g5
g4g5g5 = g4
gig5 = -g5gi
g4gig5 = g5g4gi
gig5g5 = -g5gig5
gigjg5 = g5gigj
g1g3=cp.matmul(g1,g3)
C_g5=-1 *cp.matmul(g2,g4g5)
P_matrix1=cp.zeros((4,4),dtype=complex)
P_matrix2=cp.zeros((4,4),dtype=complex)
P_matrix1=0.5 * (g0+g4)
P_matrix2=0.5 * (g0-g4)
S_DR_to_DP=(-g2+g1g3)/cp.sqrt(2)
S_trans=cp.matmul(S_DR_to_DP,S_to_DR)
S_trans_dagger=cp.transpose(cp.conj(S_trans))
P_spin_1=cp.zeros((4,4),dtype=complex)
P_spin_1[0,0]=1.0+0.0*1j
P_spin_1=cp.matmul(S_trans_dagger,cp.matmul(P_spin_1,S_trans))


#meson_list = ["g0", "g4", "g5", "g4g5", "gi", "g4gi", "gig5", "gigj"]
#meson_list = ["0Mon_I_0", "0Mon_I_1","0Mon_I_1.1"]
meson_list = ["0Mon_I_0", "0Mon_I_1"]
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
    if(tmp[0] == 'Px'):
        Px = int(tmp[1])
    if(tmp[0] == 'Py'):
        Py = int(tmp[1])
    if(tmp[0] == 'Pz'):
        Pz = int(tmp[1])
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
    return exp_diag_cupy


def VDV(t, exp_diags):
    eigvecs = readin_eigvecs(eig_dir, t, Nev, Nev1, conf_id, Nx)
    eigvecs_dagger = cp.conj(cp.transpose(eigvecs))
    n = exp_diags.shape[0]
    vdv = cp.zeros((n, Nev1, Nev1), dtype=complex)
    for i in range(n):
        vdv[i] = cp.matmul((eigvecs_dagger * exp_diags[i]), eigvecs)
    return vdv

# ------------------------------------------------------------------------------


def readin_phi_vdv(phi_dir, t_source, Nev1, conf_id):#读取t时刻的phi_vdv

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
def readin_all_phi_vdv(phi_dir, Nt, Nev1, conf_id):#读取所有t时刻的phi_vdv

    phi_vdv_all = np.zeros((Nt, Nev1, Nev1, Nev1), dtype=complex)

    for t in range(Nt):
        print("reading phi_vdv, t=%d" % t)
        f = open("%s/VVV.t%03i.Px%iPy%iPz%i.conf%s" % (phi_dir, t, Px, Py, Pz, conf_id), 'rb')
        VVV = np.fromfile(f, dtype='f8')
        Nev=int(np.cbrt(VVV.size/2))
        VVV = VVV.reshape(Nev, Nev, Nev, 2)
        VVV = VVV[..., 0] + VVV[..., 1] * 1j
        VVV = VVV[0:Nev1, 0:Nev1, 0:Nev1]
        phi_vdv_all[t,:,:,:] =VVV


    phi_vdv_cupy = cp.array(phi_vdv_all)
    return phi_vdv_cupy


# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------


if __name__ == '__main__':

    multiprocessing.set_start_method('spawn')

    st = time.time()
    exp_diags = cp.zeros((nMom, Nc*Nx**3), dtype=complex)
    exp_diags[0] = epx(Nc, Nx, Px, Py, Pz)#只有一个动量的情况，未来会拓展到多个动量
    ed = time.time()
    print("exp time used: %.3f s" % (ed-st))

    VDV_async = cp.zeros((nMom, Nt, Nev1, Nev1), dtype=complex)
    st = time.time()
    args = [t for t in range(Nt)]
    with multiprocessing.Pool(processes=nproc) as pool:
        results = []
        for t in args:
            # 使用异步方式提交任务，不阻塞主进程
            result = pool.apply_async(VDV, (t, exp_diags))
            results.append(result)

        for t, result in enumerate(results):
            VDV_async[:, t, ...] = result.get()

    ed = time.time()
    print("VDV time used: %.3f s" % (ed-st))

    # ------------------------------------------------------------------------------

#    phi_async = readin_phi_vdv(phi_dir, Nt, Nev1,conf_id)#读取所有t时刻的phi_vdv

    # ------------------------------------------------------------------------------

#    phi_vdv_source = cp.zeros((Nev1, Nev1, Nev1), dtype=complex)
    #计算VVV只能提前计算，不然时间过久

    #计算VVV的方法
    st = time.time()
    # phi_vdv_all=readin_all_phi_vdv(phi_dir, Nt, Nev1, conf_id)#读取t时刻的phi_vdv

    phi_vdv_all = cp.zeros((Nt, Nev1, Nev1,Nev1), dtype=complex)
    st = time.time()
    args = [t for t in range(Nt)]
    with multiprocessing.Pool(processes=nproc) as pool:
        results = []
        for t in args:
            # 使用异步方式提交任务，不阻塞主进程
            result = pool.apply_async(VVV_cal, (eig_dir,Nx,t,Nev1,conf_id,0,0,0))
            results.append(result)

        for t, result in enumerate(results):
            phi_vdv_all[t, ...] = result.get()

        ed = time.time()
        print("VVV time used: %.3f s" % (ed-st))

    # ------------------------------------------------------------------------------

    contrac_meson_temp = cp.zeros((len(meson_list), Nt, Nt), dtype=complex)
    #contrac_meson_temp = cp.zeros((len(meson_list),Nt,Nt), dtype=complex)

    for t_source in range(0,Nt,tstep):

        peram_u = readin_peram(peram_u_dir, conf_id, Nt, Nev, Nev1, t_source)#读取传播子
        peram_s = readin_peram(peram_s_dir, conf_id, Nt, Nev, Nev1, t_source)#读取传播子
        phi_vdv_source=phi_vdv_all[t_source]#读取t时刻的phi_vdv
        VDV_all=VDV_async[0]
        VdV_source = VDV_all[t_source]

        # CG5_peram_d_CG5 =contract("ba,tzyad,ed->tzybe", C_g5, peram_u, C_g5)#缩并d夸克
        G5_peram_s_G5 =contract("ba,tzyda,de->tzybe", -g5g5, cp.conj(peram_s), g5g5)

        for t_sink in range(Nt):#缩并得到关联函数
            phi_vdv_sink=phi_vdv_all[t_sink]#读取t时刻的phi_vdv
            peram_u_sink=peram_u[t_sink]#读取传播子
            # CG5_peram_d_CG5_sink=CG5_peram_d_CG5[t_sink]#读取传播子
            G5_peram_s_G5_sink=G5_peram_s_G5[t_sink]#读取传播子


            #缩并得到关联函数
            KppK = -\
                contract("ijk,op,gc,ab, ilcg,jmae, prdh,knbf, oqhd, ef,rq,lmn", phi_vdv_sink, VDV_all[t_sink], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, cp.conj(VdV_source.T), cp.conj(phi_vdv_source)) -\
                contract("ijk,op,gc,ab, imce,jlag, pndf,krbh, oqhd, ef,rq,lmn", phi_vdv_sink, VDV_all[t_sink], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, cp.conj(VdV_source.T), cp.conj(phi_vdv_source)) +\
                contract("ijk,op,gc,ab, imce,jlag, prdh,knbf, oqhd, ef,rq,lmn", phi_vdv_sink, VDV_all[t_sink], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, cp.conj(VdV_source.T), cp.conj(phi_vdv_source)) +\
                contract("ijk,op,gc,ab, ilcg,jmae, pndf,krbh, oqhd, ef,rq,lmn", phi_vdv_sink, VDV_all[t_sink], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, cp.conj(VdV_source.T), cp.conj(phi_vdv_source))
            KpnK = +\
                contract("ijk,op,gc,ab, incf,jrah, pmde,klbg, oqhd, ef,rq,lmn", phi_vdv_sink, VDV_all[t_sink], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, cp.conj(VdV_source.T), cp.conj(phi_vdv_source)) +\
                contract("ijk,op,gc,ab, irch,jnaf, pldg,kmbe, oqhd, ef,rq,lmn", phi_vdv_sink, VDV_all[t_sink], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, cp.conj(VdV_source.T), cp.conj(phi_vdv_source)) -\
                contract("ijk,op,gc,ab, irch,jnaf, pmde,klbg, oqhd, ef,rq,lmn", phi_vdv_sink, VDV_all[t_sink], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, cp.conj(VdV_source.T), cp.conj(phi_vdv_source)) -\
                contract("ijk,op,gc,ab, incf,jrah, pldg,kmbe, oqhd, ef,rq,lmn", phi_vdv_sink, VDV_all[t_sink], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, cp.conj(VdV_source.T), cp.conj(phi_vdv_source))
            contrac_meson_temp[0, t_sink,t_source] =  KppK - KpnK
            contrac_meson_temp[1, t_sink,t_source] =  KppK + KpnK
            # contrac_meson_temp[2, t_sink,t_source] = \
            #     -contract("ijk,op,eb, ilbe,jmad,prcf, knad,oqfc, rq,lmn", phi_vdv_sink, VDV_all[t_sink], P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, cp.conj(VdV_source.T), cp.conj(phi_vdv_source))  \
            #     -contract("ijk,op,eb, imbd,jraf,plce, knad,oqfc, rq,lmn", phi_vdv_sink, VDV_all[t_sink], P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, cp.conj(VdV_source.T), cp.conj(phi_vdv_source))  \
            #     -contract("ijk,op,eb, irbf,jlae,pmcd, knad,oqfc, rq,lmn", phi_vdv_sink, VDV_all[t_sink], P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, cp.conj(VdV_source.T), cp.conj(phi_vdv_source))  \
            #     +contract("ijk,op,eb, imbd,jlae,prcf, knad,oqfc, rq,lmn", phi_vdv_sink, VDV_all[t_sink], P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, cp.conj(VdV_source.T), cp.conj(phi_vdv_source))  \
            #     +contract("ijk,op,eb, irbf,jmad,plce, knad,oqfc, rq,lmn", phi_vdv_sink, VDV_all[t_sink], P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, cp.conj(VdV_source.T), cp.conj(phi_vdv_source))  \
            #     +contract("ijk,op,eb, ilbe,jraf,pmcd, knad,oqfc, rq,lmn", phi_vdv_sink, VDV_all[t_sink], P_matrix1, peram_u_sink, peram_u_sink, peram_u_sink, CG5_peram_d_CG5_sink, G5_peram_s_G5_sink, cp.conj(VdV_source.T), cp.conj(phi_vdv_source))

#        contrac_meson_temp[:,:,t_source] = corr(t_source, phi_vdv_all)


    ed = time.time()
    print("corr time used: %.3f s" % (ed-st))

    # ------------------------------------------------------------------------------

    corr_meson = cp.zeros((len(meson_list), 1, Nt), dtype=complex)
    st0 = time.time()
    for t_source in range(0,Nt,tstep):
        for t_sink in range(Nt):
            for i in range(len(meson_list)):
                if (t_sink < t_source):
                    contrac_meson_temp[i,t_sink, t_source]= -1 * contrac_meson_temp[i, t_sink, t_source]
                corr_meson[i, 0, (t_sink-t_source+Nt) % Nt] += contrac_meson_temp[i, t_sink, t_source]#t_sink-t_source+Nt是为了保证t_sink-t_source是正数,即编时




    ed0 = time.time()
    print("corr done, time used: %.3f s" % (ed0-st0))
    corr_meson = cp.asnumpy(corr_meson)

    st0 = time.time()
    #存数据
    for i in range(len(meson_list)):
        write_data_ascii(corr_meson[i], Nt, Nx, "%s/%02d_Px%sPy%sPz%s_conf%s.dat" %#写入的顺序有讲究没
                        (corr_dir, i, Px, Py, Pz, conf_id))
    ed0 = time.time()
    print("write data done, time used: %.3f s" % (ed0-st0))
