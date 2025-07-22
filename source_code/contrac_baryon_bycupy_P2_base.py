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
from O_group import*
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


def VDV(t, exp_diags, vectors):
    eigvecs = readin_eigvecs(eig_dir, t, Nev, Nev1, conf_id, Nx)
    eigvecs_dagger = cp.conj(cp.transpose(eigvecs))
    vdv = cp.zeros((len(vectors), Nev1, Nev1), dtype=complex)
    for i, vec in enumerate(vectors):
        vdv[i] = cp.matmul((eigvecs_dagger * exp_diags[vec]), eigvecs)
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


if __name__ == '__main__':

    #corr时间：20分钟，VVV时间22秒，VDV时间7秒
    meson_list = ["I0", "I1"]#同位旋，因为较为简单因此依旧算三个，较复杂场景会算两个
    irrep_num=5#不可约表示的数目
    #p_start_num=irrep_num 维数还行因此不选择减少计算量的方式（即source只计算p_start_num个动量）
    vectors,start_vectors = load_vectors_from_file(vector_need_dir)
    p_num= len(vectors)
    print("p_num: %d" % p_num)
#--------------------------------------------------------------------------------
    multiprocessing.set_start_method('spawn')
    st = time.time()
    exp_diags = {}   # 改为字典存储
    for vec in vectors:
        px, py, pz = vec
        exp_diags[vec] = epx(Nc, Nx, -px, -py, -pz)  # 动量方向相反

    ed = time.time()
    print("exp time used: %.3f s" % (ed-st))

    # 改为字典存储VDV_all: key=动量向量, value=时间序列的VDV矩阵
    VDV_all = {}
    for vec in vectors:
        VDV_all[vec] = cp.zeros((Nt, Nev1, Nev1), dtype=complex)

    st = time.time()
    args = [t for t in range(Nt)]
    with multiprocessing.Pool(processes=nproc) as pool:
        results = []
        for t in args:
            result = pool.apply_async(VDV, (t, exp_diags, vectors))
            results.append(result)

        for t, result in enumerate(results):
            vdv_t = result.get()  # 获取时间t的所有动量VDV
            for i, vec in enumerate(vectors):
                VDV_all[vec][t] = vdv_t[i]
    ed = time.time()
    print("VDV time used: %.3f s" % (ed-st))#得到所有的VDV矩阵(字典形式)
    # ------------------------------------------------------------------------------



    st0= time.time()
##由于显存是在不够因此选择在每个时间片计算VVV，虽然会浪费时间
    # ------------------------------------------------------------------------------
    contrac_meson_temp= cp.zeros((irrep_num, irrep_num, len(meson_list), Nt, Nt), dtype=complex)
#    为了减少重复计算量，因此先计算sink等于source的情况
    #初始化VDV与VVV
    # 处理VDV_source: 提取特定时间点所有动量的VDV并进行转置共轭
    VDV_source={}
    for vec in vectors:
        VDV_source[vec] = VDV_all[vec][0]  # 存储原始VDV矩阵
    # 计算VVV_source
    source_value = {}  # 字典: 映射动量到VVV_source
    for k in range(p_num):
        momentum = tuple(vectors[k])
        if momentum in source_value:
            continue
        vvv = VVV_cal(eig_dir, Nx, 0, Nev1, conf_id, *momentum)  # 直接调用函数
        source_value[momentum] = vvv  # 存储到字典

    for t_source in range(0, Nt, tstep):#t_source是源点时间
        st1 = time.time()
        #整理数据
        peram_u = readin_peram(peram_u_dir, conf_id, Nt, Nev1, Nev1, t_source)#读取传播子
        peram_s = readin_peram(peram_s_dir, conf_id, Nt, Nev1, Nev1, t_source)#读取传播子
        G5_peram_s_G5 =contract("ba,tzyda,de->tzybe", -g5g5, cp.conj(peram_s), g5g5)
        #基本数据
        peram_u_sink=peram_u[t_source]#读取传播子
        G5_peram_s_G5_sink=G5_peram_s_G5[t_source]#读取传播子
        #计算sink等于sink等于source的情况，计算(在此一并吧所有的都算了，包括ij和ji的情况)
        st3= time.time()
        for i in range(irrep_num):
            for j in range(i+1):
                filename = f"base_{i}{j}"
                file_path = os.path.join(base_dir, filename)
                with open(file_path, 'r') as f:
                    # 读取第一行（基底个数）
                    f.readline().strip()
                    while True:
                        # 读取系数行（格式如 "1*["）
                        coeff_line = f.readline().strip()
                        if not coeff_line:
                            break
                        # 提取数字部分（如 "1*[" -> 1）
                        if '*' in coeff_line:
                            coefficient = int(coeff_line.split('*')[0])
                        else:
                            coefficient = int(coeff_line)
                        print("coefficient: %d" % coefficient)
                        comp1 = list(map(int, f.readline().split()))
                        comp2 = list(map(int, f.readline().split()))
                        comp3 = list(map(int, f.readline().split()))
                        # 验证并读取闭括号行
                        bracket_line = f.readline().strip()
                        if bracket_line != ']':
                            print(f"警告：base文件 {filename} 缺少闭括号，跳过该基矢量")

                        #计算所需数据
                        source_vec_2 = (comp1[0], comp2[0], comp3[0])
                        source_vec_1 = (comp1[1], comp2[1], comp3[1])
                        VVV_source_1 = source_value.get(source_vec_1)
                        conj_VVV_source_1= cp.conj(VVV_source_1)
                        VDV_source_1 = VDV_source.get(source_vec_1)
                        conj_VDV_source_1 = cp.conj(VDV_source_1.transpose())
                        VVV_source_2 = source_value.get(source_vec_2)
                        conj_VVV_source_2= cp.conj(VVV_source_2)
                        VDV_source_2 = VDV_source.get(source_vec_2)
                        conj_VDV_source_2 = cp.conj(VDV_source_2.transpose())
                        #计算VVV的收缩(顺时间)(ij)
                        KnnK_0 = -\
                            contract("ijk,op,gc,ab, ilcg,jmae, prdh,knbf, oqhd, ef,rq,lmn", VVV_source_2, VDV_source_2, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) -\
                            contract("ijk,op,gc,ab, imce,jlag, pndf,krbh, oqhd, ef,rq,lmn", VVV_source_2, VDV_source_2, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) +\
                            contract("ijk,op,gc,ab, imce,jlag, prdh,knbf, oqhd, ef,rq,lmn", VVV_source_2, VDV_source_2, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) +\
                            contract("ijk,op,gc,ab, ilcg,jmae, pndf,krbh, oqhd, ef,rq,lmn", VVV_source_2, VDV_source_2, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1)
                        KnpK_0 = +\
                            contract("ijk,op,gc,ab, incf,jrah, pmde,klbg, oqhd, ef,rq,lmn", VVV_source_2, VDV_source_2, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) +\
                            contract("ijk,op,gc,ab, irch,jnaf, pldg,kmbe, oqhd, ef,rq,lmn", VVV_source_2, VDV_source_2, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) -\
                            contract("ijk,op,gc,ab, irch,jnaf, pmde,klbg, oqhd, ef,rq,lmn", VVV_source_2, VDV_source_2, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) -\
                            contract("ijk,op,gc,ab, incf,jrah, pldg,kmbe, oqhd, ef,rq,lmn", VVV_source_2, VDV_source_2, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1)
                        contrac_meson_temp[i,j,0,t_source,t_source] += coefficient * (KnnK_0 - KnpK_0) #这里的i,j是不可约表示的索引
                        contrac_meson_temp[i,j,1,t_source,t_source] += coefficient * (KnnK_0 + KnpK_0)
                        if i!= j: #如果i和j不相等，则需要计算ij和ji的情况
                        #计算VVV的收缩(顺时间)(ji)
                            KnnK_2 = -\
                                contract("ijk,op,gc,ab, ilcg,jmae, prdh,knbf, oqhd, ef,rq,lmn", VVV_source_1, VDV_source_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_2, conj_VVV_source_2) -\
                                contract("ijk,op,gc,ab, imce,jlag, pndf,krbh, oqhd, ef,rq,lmn", VVV_source_1, VDV_source_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_2, conj_VVV_source_2) +\
                                contract("ijk,op,gc,ab, imce,jlag, prdh,knbf, oqhd, ef,rq,lmn", VVV_source_1, VDV_source_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_2, conj_VVV_source_2) +\
                                contract("ijk,op,gc,ab, ilcg,jmae, pndf,krbh, oqhd, ef,rq,lmn", VVV_source_1, VDV_source_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_2, conj_VVV_source_2)
                            KnpK_2 = +\
                                contract("ijk,op,gc,ab, incf,jrah, pmde,klbg, oqhd, ef,rq,lmn", VVV_source_1, VDV_source_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_2, conj_VVV_source_2) +\
                                contract("ijk,op,gc,ab, irch,jnaf, pldg,kmbe, oqhd, ef,rq,lmn", VVV_source_1, VDV_source_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_2, conj_VVV_source_2) -\
                                contract("ijk,op,gc,ab, irch,jnaf, pmde,klbg, oqhd, ef,rq,lmn", VVV_source_1, VDV_source_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_2, conj_VVV_source_2) -\
                                contract("ijk,op,gc,ab, incf,jrah, pldg,kmbe, oqhd, ef,rq,lmn", VVV_source_1, VDV_source_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_2, conj_VVV_source_2)
                            contrac_meson_temp[j,i,0,t_source,t_source] += coefficient * (KnnK_2 - KnpK_2) #这里的i,j是不可约表示的索引
                            contrac_meson_temp[j,i,1,t_source,t_source] += coefficient * (KnnK_2 + KnpK_2)
        et3 = time.time()
        print("t_source=%d, one contract time used: %.3f s" % (t_source, et3-st3), flush=True)
        #开始计算
        for t_sink in range(Nt - 1, t_source, -tstep):#保证t_sink > t_source,循环时间片范围：71到2 (tsource+1)，步长-tstep，不包括tsource
            #基本数据
            peram_u_sink=peram_u[t_sink]#读取传播子
            G5_peram_s_G5_sink=G5_peram_s_G5[t_sink]#读取传播子
            #反向传播的传播子
            peram_u_inv=readin_peram(peram_u_dir, conf_id, Nt, Nev1, Nev1, t_sink)#读取传播子
            peram_s_inv=readin_peram(peram_s_dir, conf_id, Nt, Nev1, Nev1, t_sink)#读取传播子
            peram_u_source=peram_u[t_source]#读取传播子
            G5_peram_s_G5_source=contract("ba,tzyda,de->tzybe", -g5g5, cp.conj(peram_s_inv), g5g5)[t_source]#读取传播子
            #获取VDV_sink
            VDV_sink = {}
            for vec in vectors:
                VDV_sink[vec] = VDV_all[vec][t_sink]  # 存储原始VDV矩阵

            #计算VVV_sink
            sink_value = {}  # 字典: 映射动量到VVV_source
            for k in range(p_num):
                momentum = tuple(vectors[k])
                if momentum in sink_value:
                    continue
                vvv = VVV_cal(eig_dir, Nx, t_sink, Nev1, conf_id, *momentum)  # 直接调用函数
                sink_value[momentum] = vvv  # 存储到字典
#-------------------------------------------------------------------------------------------------
            #计算关联函数
#-------------------------------------------------------------------------------------------------
            #初始化收缩函数
            st2 = time.time()
            #计算(在此一并吧所有的都算了，包括顺时间和逆时间，以及ij和ji的情况)
            for i in range(irrep_num):
                for j in range(i+1):
                    filename = f"base_{i}{j}"
                    file_path = os.path.join(base_dir, filename)
                    with open(file_path, 'r') as f:
                        # 读取第一行（基底个数）
                        base_count = f.readline().strip()
                        while True:
                            # 读取系数行（格式如 "1*["）
                            coeff_line = f.readline().strip()
                            if not coeff_line:
                                break
                            # 提取数字部分（如 "1*[" -> 1）
                            if '*' in coeff_line:
                                coefficient = int(coeff_line.split('*')[0])
                            else:
                                coefficient = int(coeff_line)
                            # 读取向量分量行
                            comp1 = list(map(int, f.readline().split()))
                            comp2 = list(map(int, f.readline().split()))
                            comp3 = list(map(int, f.readline().split()))
                            # 读取闭括号行 "]"并跳过
                            bracket_line = f.readline().strip()

                            sink_vec_1 = (comp1[0], comp2[0],comp3[0])
                            source_vec_1 = (comp1[1], comp2[1], comp3[1])
                            VVV_source_1 = source_value.get(source_vec_1)
                            VVV_sink_1 = sink_value.get(sink_vec_1)
                            conj_VVV_source_1= cp.conj(VVV_source_1)
                            conj_VVV_sink_1= cp.conj(VVV_sink_1)
                            VDV_source_1 = VDV_source.get(source_vec_1)
                            VDV_sink_1 = VDV_sink.get(sink_vec_1)
                            conj_VDV_source_1 = cp.conj(VDV_source_1.transpose())
                            conj_VDV_sink_1 = cp.conj(VDV_sink_1.transpose())

                            sink_vec_2 = (comp1[1], comp2[1],comp3[1])
                            source_vec_2 = (comp1[0], comp2[0], comp3[0])
                            VVV_source_2 = source_value.get(source_vec_2)
                            VVV_sink_2 = sink_value.get(sink_vec_2)
                            conj_VVV_source_2= cp.conj(VVV_source_2)
                            conj_VVV_sink_2= cp.conj(VVV_sink_2)
                            VDV_source_2 = VDV_source.get(source_vec_2)
                            VDV_sink_2 = VDV_sink.get(sink_vec_2)
                            conj_VDV_source_2 = cp.conj(VDV_source_2.transpose())
                            conj_VDV_sink_2 = cp.conj(VDV_sink_2.transpose())

                            #计算VVV的收缩(顺时间)(ij)
                            KnnK_0 = -\
                contract("ijk,op,gc,ab, ilcg,jmae, prdh,knbf, oqhd, ef,rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) -\
                contract("ijk,op,gc,ab, imce,jlag, pndf,krbh, oqhd, ef,rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) +\
                contract("ijk,op,gc,ab, imce,jlag, prdh,knbf, oqhd, ef,rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) +\
                contract("ijk,op,gc,ab, ilcg,jmae, pndf,krbh, oqhd, ef,rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1)
                            KnpK_0 = +\
                contract("ijk,op,gc,ab, incf,jrah, pmde,klbg, oqhd, ef,rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) +\
                contract("ijk,op,gc,ab, irch,jnaf, pldg,kmbe, oqhd, ef,rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) -\
                contract("ijk,op,gc,ab, irch,jnaf, pmde,klbg, oqhd, ef,rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1) -\
                contract("ijk,op,gc,ab, incf,jrah, pldg,kmbe, oqhd, ef,rq,lmn", VVV_sink_1, VDV_sink_1, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_1, conj_VVV_source_1)
                            contrac_meson_temp[i,j,0,t_sink,t_source] += coefficient * (KnnK_0 - KnpK_0) #这里的i,j是不可约表示的索引
                            contrac_meson_temp[i,j,1,t_sink,t_source] += coefficient * (KnnK_0 + KnpK_0)
                            #计算VVV的收缩(逆时间)(ij)
                            KnnK_1 = -\
                contract("ijk,op,gc,ab, ilcg,jmae, prdh,knbf, oqhd, ef,rq,lmn", VVV_source_2, VDV_source_2, P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_2, conj_VVV_sink_2) -\
                contract("ijk,op,gc,ab, imce,jlag, pndf,krbh, oqhd, ef,rq,lmn", VVV_source_2, VDV_source_2, P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_2, conj_VVV_sink_2) +\
                contract("ijk,op,gc,ab, imce,jlag, prdh,knbf, oqhd, ef,rq,lmn", VVV_source_2, VDV_source_2, P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_2, conj_VVV_sink_2) +\
                contract("ijk,op,gc,ab, ilcg,jmae, pndf,krbh, oqhd, ef,rq,lmn", VVV_source_2, VDV_source_2, P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_2, conj_VVV_sink_2)
                            KnpK_1 = +\
                contract("ijk,op,gc,ab, incf,jrah, pmde,klbg, oqhd, ef,rq,lmn", VVV_source_2, VDV_source_2, P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_2, conj_VVV_sink_2) +\
                contract("ijk,op,gc,ab, irch,jnaf, pldg,kmbe, oqhd, ef,rq,lmn", VVV_source_2, VDV_source_2, P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_2, conj_VVV_sink_2) -\
                contract("ijk,op,gc,ab, irch,jnaf, pmde,klbg, oqhd, ef,rq,lmn", VVV_source_2, VDV_source_2, P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_2, conj_VVV_sink_2) -\
                contract("ijk,op,gc,ab, incf,jrah, pldg,kmbe, oqhd, ef,rq,lmn", VVV_source_2, VDV_source_2, P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_2, conj_VVV_sink_2)
                            contrac_meson_temp[i,j,0,t_source,t_sink] += coefficient * (KnnK_1 - KnpK_1) #这里的i,j是不可约表示的索引
                            contrac_meson_temp[i,j,1,t_source,t_sink] += coefficient * (KnnK_1 + KnpK_1)

                            if i!= j: #如果i和j不相等，则需要计算ij和ji的情况
                            #计算VVV的收缩(顺时间)(ji)
                                KnnK_2 = -\
                                    contract("ijk,op,gc,ab, ilcg,jmae, prdh,knbf, oqhd, ef,rq,lmn", VVV_sink_2, VDV_sink_2, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_2, conj_VVV_source_2) -\
                                    contract("ijk,op,gc,ab, imce,jlag, pndf,krbh, oqhd, ef,rq,lmn", VVV_sink_2, VDV_sink_2, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_2, conj_VVV_source_2) +\
                                    contract("ijk,op,gc,ab, imce,jlag, prdh,knbf, oqhd, ef,rq,lmn", VVV_sink_2, VDV_sink_2, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_2, conj_VVV_source_2) +\
                                    contract("ijk,op,gc,ab, ilcg,jmae, pndf,krbh, oqhd, ef,rq,lmn", VVV_sink_2, VDV_sink_2, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_2, conj_VVV_source_2)
                                KnpK_2 = +\
                                    contract("ijk,op,gc,ab, incf,jrah, pmde,klbg, oqhd, ef,rq,lmn", VVV_sink_2, VDV_sink_2, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_2, conj_VVV_source_2) +\
                                    contract("ijk,op,gc,ab, irch,jnaf, pldg,kmbe, oqhd, ef,rq,lmn", VVV_sink_2, VDV_sink_2, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_2, conj_VVV_source_2) -\
                                    contract("ijk,op,gc,ab, irch,jnaf, pmde,klbg, oqhd, ef,rq,lmn", VVV_sink_2, VDV_sink_2, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_2, conj_VVV_source_2) -\
                                    contract("ijk,op,gc,ab, incf,jrah, pldg,kmbe, oqhd, ef,rq,lmn", VVV_sink_2, VDV_sink_2, P_matrix1, C_g5, peram_u_sink, peram_u_sink, peram_u_sink, peram_u_sink, G5_peram_s_G5_sink, C_g5, conj_VDV_source_2, conj_VVV_source_2)
                                contrac_meson_temp[j,i,0,t_sink,t_source] += coefficient * (KnnK_2 - KnpK_2) #这里的i,j是不可约表示的索引
                                contrac_meson_temp[j,i,1,t_sink,t_source] += coefficient * (KnnK_2 + KnpK_2)
                            #计算VVV的收缩(逆时间)(ji)
                                KnnK_3 = -\
                                    contract("ijk,op,gc,ab, ilcg,jmae, prdh,knbf, oqhd, ef,rq,lmn", VVV_source_1, VDV_source_1, P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_1, conj_VVV_sink_1) -\
                                    contract("ijk,op,gc,ab, imce,jlag, pndf,krbh, oqhd, ef,rq,lmn", VVV_source_1, VDV_source_1, P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_1, conj_VVV_sink_1) +\
                                    contract("ijk,op,gc,ab, imce,jlag, prdh,knbf, oqhd, ef,rq,lmn", VVV_source_1, VDV_source_1, P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_1, conj_VVV_sink_1) +\
                                    contract("ijk,op,gc,ab, ilcg,jmae, pndf,krbh, oqhd, ef,rq,lmn", VVV_source_1, VDV_source_1, P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_1, conj_VVV_sink_1)
                                KnpK_3 = +\
                                    contract("ijk,op,gc,ab, incf,jrah, pmde,klbg, oqhd, ef,rq,lmn", VVV_source_1, VDV_source_1, P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_1, conj_VVV_sink_1) +\
                                    contract("ijk,op,gc,ab, irch,jnaf, pldg,kmbe, oqhd, ef,rq,lmn", VVV_source_1, VDV_source_1, P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_1, conj_VVV_sink_1) -\
                                    contract("ijk,op,gc,ab, irch,jnaf, pmde,klbg, oqhd, ef,rq,lmn", VVV_source_1, VDV_source_1, P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_1, conj_VVV_sink_1) -\
                                    contract("ijk,op,gc,ab, incf,jrah, pldg,kmbe, oqhd, ef,rq,lmn", VVV_source_1, VDV_source_1, P_matrix1, C_g5, peram_u_source, peram_u_source, peram_u_source, peram_u_source, G5_peram_s_G5_source, C_g5, conj_VDV_sink_1, conj_VVV_sink_1)
                                contrac_meson_temp[j,i,0,t_source,t_sink] += coefficient * (KnnK_3 - KnpK_3) #这里的i,j是不可约表示的索引
                                contrac_meson_temp[j,i,1,t_source,t_sink] += coefficient * (KnnK_3 + KnpK_3)
            ed2 = time.time()
            print("contract time used: %.3f s" % (ed2-st2),flush=True)
        if t_sink == t_source + 1:
            print("ture")
            VDV_source = {vec: cp.copy(array) for vec, array in VDV_sink.items()}
            source_value = {vec: cp.copy(array) for vec, array in sink_value.items()}  # 更新VVV_source
        else:
            print("false")
        ed1 = time.time()
        print("t_source: %d, time used: %.3f s" % (t_source, ed1-st1), flush=True)


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
            corr_meson_wirte=corr_meson[i, j, :,:,:]
            for k in range(len(meson_list)):
                write_data_ascii(corr_meson_wirte[k], Nt, Nx, "%s/%02d_P%s_conf%s_%d%d.dat" %#写入的顺序有讲究没
                        (corr_dir, k, 2, conf_id,i,j))
    ed0 = time.time()
    print("write data done, time used: %.3f s" % (ed0-st0))
