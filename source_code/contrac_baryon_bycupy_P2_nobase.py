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
from O_group.compute_matrix import *
from O_group.generte_Ogrpup import load_operators
from O_group import*
from opt_einsum import contract
import time
import multiprocessing
import sys
import copy
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
g1g3=cp.matmul(g1, g3)
P_matrix1=0.5 * (g0+g4)
P_matrix2=0.5 * (g0-g4)



#meson_list = ["g0", "g4", "g5", "g4g5", "gi", "g4gi", "gig5", "gigj"]
meson_list = ["I_0", "I_1",]
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
    if(tmp[0] == 'base_dir'):
        base_dir = tmp[1]
    if(tmp[0] == 'vector_need_dir'):
        vector_need_dir = tmp[1]
    if(tmp[0] == 'all_operator_dir'):
        all_operator_dir = tmp[1]
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
    vdv_list = []
    for exp_diag in exp_diags:
        vdv = cp.matmul((eigvecs_dagger * exp_diag), eigvecs)
        vdv_list.append(vdv)
    return vdv_list

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

#---------------------------------------------------------------
# 定义计算函数
def compute_vvv(args):
    i, irrep_idx, mom_idx, vec = args
    t_val = Nt - 1 - i  # 修正时间计算
    return (i, irrep_idx, mom_idx, VVV_cal(eig_dir, Nx, t_val, Nev1, conf_id, *vec))
# ------------------------------------------------------------------------------
# def readin_all_phi_vdv(phi_dir, Nt, Nev1, conf_id, p_num):#读取所有t时刻的phi_vdv

#     phi_vdv_all = np.zeros((p_num, Nt, Nev1, Nev1, Nev1), dtype=complex)
#     for k in range(p_num):
#         for t in range(Nt):
#             print("reading phi_vdv, t=%d" % t)
#             f = open("%s/VVV.t%03i.Px%iPy%iPz%i.conf%s" % (phi_dir, t, Px[k], Py[k], Pz[k], conf_id), 'rb')
#             VVV = np.fromfile(f, dtype='f8')
#             Nev=int(np.cbrt(VVV.size/2))
#             VVV = VVV.reshape(Nev, Nev, Nev, 2)
#             VVV = VVV[..., 0] + VVV[..., 1] * 1j
#             VVV = VVV[0:Nev1, 0:Nev1, 0:Nev1]
#             phi_vdv_all[k,t,:,:,:] =VVV


#     phi_vdv_cupy = cp.array(phi_vdv_all)
#     del phi_vdv_all
#     return phi_vdv_cupy


# ------------------------------------------------------------------------------
#计算逻辑：因为为S波，因此角动量和自旋可以分开算。并且在计算角动量部分时可以运用对称性来计算，如s波本质只有+-z，x方向
#为了节约内存，在此选择使用现算的方式来计算phi_vdv，以及在每个时间片现用vvv
#但是在转动时，关联函数可能会出一个相位产生干涉，因此选择先计算c62中关联函数再组合
#三个不可约表示一共要计算
# ------------------------------------------------------------------------------
#因为每次时间最末尾的t_sink都要计算，因此可以计算最后十个VVV，然后这样可以节约很多时间

if __name__ == '__main__':

    #corr时间：20分钟，VVV时间22秒，VDV时间7秒
    meson_list = ["I0", "I1"]#同位旋，因为较为简单因此依旧算三个，较复杂场景会算两个
    #p_start_num=irrep_num 维数还行因此不选择减少计算量的方式（即source只计算p_start_num个动量）

    if not os.path.exists(all_operator_dir):
        raise FileNotFoundError(f"文件 {all_operator_dir} 不存在")
    vectors = load_operators(all_operator_dir)
    p_num= 0  # 每个不可约表示的动量数量
    print(f"从文件加载了 {len(vectors)} 组算子")
    irrep_num = len(vectors)  # 不可约表示的数量
    for i, ops in enumerate(vectors):
        print(f"第{i}组包含 {len(ops)} 个算子")
        p_num   += len(ops)  # 每个不可约表示的动量数量
    print(f"总共有 {p_num} 个动量")
#--------------------------------------------------------------------------------
    multiprocessing.set_start_method('spawn')
    st = time.time()
    # 初始化exp_diags为二维列表
    exp_diags = []
    for irrep_idx in range(irrep_num):
        exp_diags.append([])
        for vec in vectors[irrep_idx]:
            px, py, pz = vec
            exp_diags[irrep_idx].append(epx(Nc, Nx, -px, -py, -pz))
    ed = time.time()
    print("exp time used: %.3f s" % (ed-st))

    # 初始化VDV_all为三维cupy张量 [irrep, mom, t, nev1, nev1]
    VDV_all = []
    for irrep_idx in range(irrep_num):
        num_mom = len(vectors[irrep_idx])
        # 预分配连续内存空间
        VDV_all.append(cp.zeros((num_mom, Nt, Nev1, Nev1), dtype=complex))

    st = time.time()
    args = [t for t in range(Nt)]
    with multiprocessing.Pool(processes=nproc) as pool:
        results = []
        for t in args:
            # 收集所有动量对应的exp_diag
            exp_diags_list = []
            for irrep_idx in range(irrep_num):
                for vec in vectors[irrep_idx]:
                    exp_diags_list.append(exp_diags[irrep_idx][vectors[irrep_idx].index(vec)])
            result = pool.apply_async(VDV, (t, exp_diags_list))
            results.append(result)

        # 初始化索引计数器
        index_counter = [0] * irrep_num
        for t, result in enumerate(results):
            vdv_t_list = result.get()  # 获取时间t的所有动量VDV(列表形式)
            # 重置每个irrep的索引
            for irrep_idx in range(irrep_num):
                index_counter[irrep_idx] = 0
            # 将结果分配到VDV_all
            for vdv in vdv_t_list:
                # 找到当前动量属于哪个irrep
                found = False
                for irrep_idx in range(irrep_num):
                    if index_counter[irrep_idx] < len(vectors[irrep_idx]):
                        # 分配并递增索引
                        VDV_all[irrep_idx][index_counter[irrep_idx], t] = vdv
                        index_counter[irrep_idx] += 1
                        found = True
                        break
                if not found:
                    raise RuntimeError("VDV分配失败：未找到匹配的irrep")
    ed = time.time()
    print("VDV_all time used: %.3f s" % (ed-st))#得到所有的VDV矩阵(三维张量形式)
    # ------------------------------------------------------------------------------
    st2= time.time()
    # #由于显存限制，不能处理全部的VVV,但是可以预处理一部分VVV
    T_VVV = 15  # 预处理的VVV数量
    VVV_pre = []
    for pre_idx in range(T_VVV):  # 从Nt-1到Nt-T_VVV
        VVV_pre.append([])  # 初始化VVV_pre的每个不可约表示
        for irrep_idx in range(irrep_num):
            num_mom = len(vectors[irrep_idx])
            # 预分配连续内存空间
            VVV_pre[pre_idx].append(cp.zeros((num_mom, Nev1, Nev1, Nev1), dtype=complex))
            for mom_idx, vec in enumerate(vectors[irrep_idx]):
                vvv = VVV_cal(eig_dir, Nx, Nt-1-pre_idx, Nev1, conf_id, *vec)
                VVV_pre[pre_idx][irrep_idx][mom_idx] = vvv
    #由于显存限制，不能处理全部的VVV,但是可以预处理一部分VVV
    # VVV_pre = []
    # # 预分配内存
    # for i in range(T_VVV):
    #     VVV_pre.append([])
    #     for irrep_idx in range(irrep_num):
    #         num_mom = len(vectors[irrep_idx])
    #         VVV_pre[i].append(cp.zeros((num_mom, Nev1, Nev1, Nev1), dtype=complex))

    # # 准备参数列表 (i, irrep_idx, mom_idx, vec)
    # args_list = []
    # for i in range(T_VVV):
    #     for irrep_idx in range(irrep_num):
    #         for mom_idx, vec in enumerate(vectors[irrep_idx]):
    #             args_list.append((i, irrep_idx, mom_idx, vec))
    # # 使用进程池并行计算
    # with multiprocessing.Pool(processes=nproc) as pool:
    #     results = pool.map(compute_vvv, args_list)
    # # 填充结果到预分配数组
    # for (i, irrep_idx, mom_idx, vvv) in results:
    #     VVV_pre[i][irrep_idx][mom_idx] = vvv
    ed2 = time.time()
    print("VVV_pre time used: %.3f s" % (ed2-st2))#得到所有的VVV矩阵(四维张量形式)
    #-------------------------------------------------------------------------------
    st0= time.time()
    ##由于显存是在不够因此选择在每个时间片计算VVV，虽然会浪费时间
    # ------------------------------------------------------------------------------
    contrac_meson_temp= cp.zeros((irrep_num, irrep_num, len(meson_list), Nt, Nt), dtype=complex)
    #为了减少重复计算量，因此先计算sink等于source的情况
    #初始化VDV与VVV
    # 处理VDV_source: 提取特定时间点所有动量的VDV并进行转置共轭
    VDV_source=[]
    for irrep_idx in range(irrep_num):
        num_mom = len(vectors[irrep_idx])
        VDV_source.append(cp.zeros((num_mom, Nev1, Nev1), dtype=complex))
        VDV_source[irrep_idx] = VDV_all[irrep_idx][:,0,...]  # 存储原始VDV矩阵
    # 一样是在每个时间片计算VVV
    # 初始化VVV_source为四维张量 [irrep, mom, nev1, nev1, nev1]
    st = time.time()
    VVV_source = []
    for irrep_idx in range(irrep_num):
        num_mom = len(vectors[irrep_idx])
        # 预分配连续内存空间
        VVV_source.append(cp.zeros((num_mom, Nev1, Nev1, Nev1), dtype=complex))
        for mom_idx, vec in enumerate(vectors[irrep_idx]):
            vvv = VVV_cal(eig_dir, Nx, 0, Nev1, conf_id, *vec)
            VVV_source[irrep_idx][mom_idx] = vvv

    #整理VDV与VVV
    conj_VDV_source_T=[]
    conj_VVV_source=[]
    for irrep_idx in range(irrep_num):
        conj_VVV_source.append(cp.conj(VVV_source[irrep_idx]))  # 共轭
        conj_VDV_source_T.append(cp.conj(cp.transpose(VDV_source[irrep_idx], axes=(0, 2, 1))))  # 转置共轭
    ed = time.time()
    print("VVV time used: %.3f s" % (ed-st))#得到所有的VVV矩阵(四维张量形式)

    # ------------------------------------------------------------------------------
    for t_source in range(0, Nt, tstep):#t_source是源点时间
        #整理数据
        pre_idx=0#初始化预处理
        peram_u = readin_peram(peram_u_dir, conf_id, Nt, Nev1, Nev1, t_source)#读取传播子
        peram_s = readin_peram(peram_s_dir, conf_id, Nt, Nev1, Nev1, t_source)#读取传播子
        G5_peram_s_G5 =contract("ba,tzyda,de->tzybe", -g5g5, cp.conj(peram_s), g5g5)
        peram_u_sink=peram_u[t_source]#读取传播子
        G5_peram_s_G5_sink=G5_peram_s_G5[t_source]#读取传播子

        #计算sink等于sink等于source的情况，计算(在此一并吧所有的都算了，包括ij和ji的情况)
        st3= time.time()
        for i in range(irrep_num):
            for j in range(irrep_num):
                if (i == j == 1) or (i == 1 and j == 0) or (i == 0 and j == 1):  # 只计算I0和I1的情况
                    continue
                #计算VVV的收缩(顺时间)(ij)
                KnnK_0 = -\
                    contract("xijk,xop,gc,ab, ilcg,jmae, prdh,knbf, oqhd, ef,yrq,ylmn", VVV_source[i], VDV_source[i], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_T[j], conj_VVV_source[j]) -\
                    contract("xijk,xop,gc,ab, imce,jlag, pndf,krbh, oqhd, ef,yrq,ylmn", VVV_source[i], VDV_source[i], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_T[j], conj_VVV_source[j]) +\
                    contract("xijk,xop,gc,ab, imce,jlag, prdh,knbf, oqhd, ef,yrq,ylmn", VVV_source[i], VDV_source[i], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_T[j], conj_VVV_source[j]) +\
                    contract("xijk,xop,gc,ab, ilcg,jmae, pndf,krbh, oqhd, ef,yrq,ylmn", VVV_source[i], VDV_source[i], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_T[j], conj_VVV_source[j])
                KnpK_0 = +\
                    contract("xijk,xop,gc,ab, incf,jrah, pmde,klbg, oqhd, ef,yrq,ylmn", VVV_source[i], VDV_source[i], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_T[j], conj_VVV_source[j]) +\
                    contract("xijk,xop,gc,ab, irch,jnaf, pldg,kmbe, oqhd, ef,yrq,ylmn", VVV_source[i], VDV_source[i], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_T[j], conj_VVV_source[j]) -\
                    contract("xijk,xop,gc,ab, irch,jnaf, pmde,klbg, oqhd, ef,yrq,ylmn", VVV_source[i], VDV_source[i], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_T[j], conj_VVV_source[j]) -\
                    contract("xijk,xop,gc,ab, incf,jrah, pldg,kmbe, oqhd, ef,yrq,ylmn", VVV_source[i], VDV_source[i], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_T[j], conj_VVV_source[j])
                contrac_meson_temp[i,j,0,t_source,t_source] = KnnK_0 - KnpK_0 #这里的i,j是不可约表示的索引
                contrac_meson_temp[i,j,1,t_source,t_source] = KnnK_0 + KnpK_0
        et3 = time.time()
        print("t_source=%d, one contract time used: %.3f s" % (t_source, et3-st3), flush=True)


        #开始计算
        for t_sink in range(Nt - 1, t_source, -tstep):#保证t_sink > t_source,循环时间片范围：71到2 (tsource+1)，步长-tstep，不包括tsource
            st1= time.time()
            #基本数据
            peram_u_sink=peram_u[t_sink]#读取传播子
            G5_peram_s_G5_sink=G5_peram_s_G5[t_sink]#读取传播子
            #反向传播基本数据
            peram_u_inv=readin_peram(peram_u_dir, conf_id, Nt, Nev1, Nev1, t_sink)#读取传播子
            peram_s_inv=readin_peram(peram_s_dir, conf_id, Nt, Nev1, Nev1, t_sink)#读取传播子
            peram_u_source=peram_u[t_source]#读取传播子
            G5_peram_s_G5_source=contract("ba,tzyda,de->tzybe", -g5g5, cp.conj(peram_s_inv), g5g5)[t_source]#读取传播子

            # 获取VDV_sink (二维列表),计算VVV_sink (二维列表)
            VDV_sink=[]
            for irrep_idx in range(irrep_num):
                VDV_sink.append(cp.zeros((len(vectors[irrep_idx]), Nev1, Nev1), dtype=complex))
                VDV_sink[irrep_idx] = VDV_all[irrep_idx][:,t_sink,...]  # 存储原始VDV矩阵

            if pre_idx<T_VVV:
                VVV_sink = VVV_pre[pre_idx]  # 使用预处理的VVV
                pre_idx += 1  # 更新预处理索引
                print("Using precomputed VVV for t_sink=%d, index=%d" % (t_sink, pre_idx-1), flush=True)
            else:
                print("Computing VVV for t_sink=%d from scratch" % (t_sink), flush=True)
                VVV_sink = []
                for irrep_idx in range(irrep_num):
                    num_mom = len(vectors[irrep_idx])
                    # 预分配连续内存空间
                    VVV_sink.append(cp.zeros((num_mom, Nev1, Nev1, Nev1), dtype=complex))
                    for mom_idx, vec in enumerate(vectors[irrep_idx]):
                        vvv = VVV_cal(eig_dir, Nx, t_sink, Nev1, conf_id, *vec)
                        VVV_sink[irrep_idx][mom_idx] = vvv
            # 整理VDV与VVV_sink
            conj_VDV_sink_T=[]
            conj_VVV_sink=[]
            for irrep_idx in range(irrep_num):
                conj_VVV_sink.append(cp.conj(VVV_sink[irrep_idx]))  # 共轭
                conj_VDV_sink_T.append(cp.conj(cp.transpose(VDV_sink[irrep_idx], axes=(0, 2, 1))))  # 转置共轭
#-------------------------------------------------------------------------------------------------
            #计算关联函数
#-------------------------------------------------------------------------------------------------
            #初始化收缩函数

            #计算(在此一并吧所有的都算了，包括顺时间和逆时间
            for i in range(irrep_num):
                for j in range(irrep_num):
                    if (i == j == 1) or (i == 1 and j == 0) or (i == 0 and j == 1):
                        continue
                    #计算VVV的收缩(顺时间)(ij)
                    KnnK_0 = -\
                        contract("xijk,xop,gc,ab, ilcg,jmae, prdh,knbf, oqhd, ef,yrq,ylmn", VVV_sink[i], VDV_sink[i], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_T[j], conj_VVV_source[j]) -\
                        contract("xijk,xop,gc,ab, imce,jlag, pndf,krbh, oqhd, ef,yrq,ylmn", VVV_sink[i], VDV_sink[i], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_T[j], conj_VVV_source[j]) +\
                        contract("xijk,xop,gc,ab, imce,jlag, prdh,knbf, oqhd, ef,yrq,ylmn", VVV_sink[i], VDV_sink[i], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_T[j], conj_VVV_source[j]) +\
                        contract("xijk,xop,gc,ab, ilcg,jmae, pndf,krbh, oqhd, ef,yrq,ylmn", VVV_sink[i], VDV_sink[i], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_T[j], conj_VVV_source[j])
                    KnpK_0 = +\
                        contract("xijk,xop,gc,ab, incf,jrah, pmde,klbg, oqhd, ef,yrq,ylmn", VVV_sink[i], VDV_sink[i], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_T[j], conj_VVV_source[j]) +\
                        contract("xijk,xop,gc,ab, irch,jnaf, pldg,kmbe, oqhd, ef,yrq,ylmn", VVV_sink[i], VDV_sink[i], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_T[j], conj_VVV_source[j]) -\
                        contract("xijk,xop,gc,ab, irch,jnaf, pmde,klbg, oqhd, ef,yrq,ylmn", VVV_sink[i], VDV_sink[i], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_T[j], conj_VVV_source[j]) -\
                        contract("xijk,xop,gc,ab, incf,jrah, pldg,kmbe, oqhd, ef,yrq,ylmn", VVV_sink[i], VDV_sink[i], P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_T[j], conj_VVV_source[j])
                    contrac_meson_temp[i,j,0,t_sink,t_source] = KnnK_0 - KnpK_0 #这里的i,j是不可约表示的索引
                    contrac_meson_temp[i,j,1,t_sink,t_source] = KnnK_0 + KnpK_0
                    #计算VVV的收缩(逆时间)(ij)
                    KnnK_1 = -\
                        contract("xijk,xop,gc,ab, ilcg,jmae, prdh,knbf, oqhd, ef,yrq,ylmn", VVV_source[i], VDV_source[i], P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_T[j], conj_VVV_sink[j]) -\
                        contract("xijk,xop,gc,ab, imce,jlag, pndf,krbh, oqhd, ef,yrq,ylmn", VVV_source[i], VDV_source[i], P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_T[j], conj_VVV_sink[j]) +\
                        contract("xijk,xop,gc,ab, imce,jlag, prdh,knbf, oqhd, ef,yrq,ylmn", VVV_source[i], VDV_source[i], P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_T[j], conj_VVV_sink[j]) +\
                        contract("xijk,xop,gc,ab, ilcg,jmae, pndf,krbh, oqhd, ef,yrq,ylmn", VVV_source[i], VDV_source[i], P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_T[j], conj_VVV_sink[j])
                    KnpK_1 = +\
                        contract("xijk,xop,gc,ab, incf,jrah, pmde,klbg, oqhd, ef,yrq,ylmn", VVV_source[i], VDV_source[i], P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_T[j], conj_VVV_sink[j]) +\
                        contract("xijk,xop,gc,ab, irch,jnaf, pldg,kmbe, oqhd, ef,yrq,ylmn", VVV_source[i], VDV_source[i], P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_T[j], conj_VVV_sink[j]) -\
                        contract("xijk,xop,gc,ab, irch,jnaf, pmde,klbg, oqhd, ef,yrq,ylmn", VVV_source[i], VDV_source[i], P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_T[j], conj_VVV_sink[j]) -\
                        contract("xijk,xop,gc,ab, incf,jrah, pldg,kmbe, oqhd, ef,yrq,ylmn", VVV_source[i], VDV_source[i], P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_T[j], conj_VVV_sink[j])
                    contrac_meson_temp[i,j,0,t_source,t_sink] = KnnK_1 - KnpK_1 #这里的i,j是不可约表示的索引
                    contrac_meson_temp[i,j,1,t_source,t_sink] = KnnK_1 + KnpK_1
            et1 = time.time()
            print("t_source=%d, t_sink=%d, contract time used: %.3f s" % (t_source, t_sink, et1-st1), flush=True)

        if t_sink == t_source + 1:
            print("ture")
            VDV_source = copy.deepcopy(VDV_sink)  # 更新VDV_source
            VVV_source = copy.deepcopy(VVV_sink)  # 更新VVV_source
            conj_VDV_source_T = copy.deepcopy(conj_VDV_sink_T)  # 更新转置共轭VDV_source
            conj_VVV_source = copy.deepcopy(conj_VVV_sink)  # 更新共轭VVV_source
        else:
            print("false")




#        contrac_meson_temp[:,:,t_source] = corr(t_source, phi_vdv_all)


    ed0 = time.time()
    print("corr time used: %.3f s" % (ed0-st0))

# ------------------------------------------------------------------------------

    #存储关联函数
    corr_meson = cp.zeros((irrep_num, irrep_num, len(meson_list), 1, Nt,), dtype=complex)#真实的关联函数
    for t_source in range(0,Nt,tstep):
        for t_sink in range(Nt):
            if (t_sink < t_source):
                contrac_meson_temp[:,:,:,t_sink, t_source]= -1 * contrac_meson_temp[:,:,:, t_sink, t_source]
            corr_meson[:,:,:, 0, (t_sink-t_source+Nt) % Nt] += contrac_meson_temp[:,:,:, t_sink, t_source]#t_sink-t_source+Nt是为了保证t_sink-t_source是正数,即编时
    corr_meson = cp.asnumpy(corr_meson)

    st0 = time.time()
    corr_meson_wirte=np.zeros((len(meson_list), 1, Nt),dtype=complex)
    #存数据
    for i in range (irrep_num):
        for j in range(irrep_num):
            if (i == j == 1) or (i == 1 and j == 0) or (i == 0 and j == 1):  # 只计算I0和I1的情况
                continue
            corr_meson_wirte=corr_meson[i, j, :,:,:]
            for k in range(len(meson_list)):
                write_data_ascii(corr_meson_wirte[k], Nt, Nx, "%s/%02d_P%s_conf%s_%d%d.dat" %#写入的顺序有讲究没
                        (corr_dir, k, 2, conf_id,i,j))
    ed0 = time.time()
    print("write data done, time used: %.3f s" % (ed0-st0))
