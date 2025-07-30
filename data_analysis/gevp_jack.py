import numpy as np
# from scipy.linalg import eigvalsh,eigvals
from scipy import linalg
import matplotlib.pyplot as plt


# def GEVP(params, corrmat, corrmat_one, t0, tref, key):
def GEVP(T, corrmat, t0, tref):
    # corrmat = corrmat.real
    # corrmat_one = corrmat_one.real
    Nstates = corrmat.shape[1]
    relen = corrmat.shape[0]
    Gdata = np.zeros([relen,Nstates,T],dtype=complex)
    vec = np.zeros([relen,Nstates,Nstates,T],dtype=complex)
    # Gdata_one = np.zeros([Nstates,T],dtype=complex)

    for i in range(relen):
        [Gseries, refv] = linalg.eig(corrmat[i,:,:,tref],corrmat[i,:,:,t0])
        refv = refv[:,np.argsort(Gseries.real)]
        if tref > t0:
            refv = refv[:,::-1]

        for t in range(T):
            [Gseries, v] = linalg.eig(corrmat[i,:,:,t],corrmat[i,:,:,t0])
            stateargs = np.zeros(Nstates,dtype=int)
            for refstate in range(Nstates):
                stateoverlaps = np.zeros(Nstates,dtype=complex)
                for state in range(Nstates):
                    stateoverlaps[state] = np.dot(np.conjugate(refv[:,refstate]),np.dot(corrmat[i,:,:,t0],v[:,state]))
                stateargs[refstate] = np.argmax(np.abs(stateoverlaps))

            #匹配排序法
            Gdata[i,:,t] = Gseries[stateargs]
            vec[i,:,:,t]= v[:,stateargs]

    # [Gseries, refv] = linalg.eig(corrmat_one[:,:,tref],corrmat_one[:,:,t0])
    # refv = refv[:,np.argsort(Gseries)]
    # if tref > t0:
    #     refv = refv[:,::-1]
    # for t in range(T):
    #     [Gseries, v] = linalg.eig(corrmat_one[:,:,t],corrmat_one[:,:,t0])
    #     stateargs = np.zeros(Nstates,dtype=int)
    #     for refstate in range(Nstates):
    #         stateoverlaps = np.zeros(Nstates,dtype=complex)
    #         for state in range(Nstates):
    #             stateoverlaps[state] = np.dot(np.conjugate(refv[:,refstate]),np.dot(corrmat_one[:,:,t0],v[:,state]))
    #         stateargs[refstate] = np.argmax(np.abs(stateoverlaps))
    #     Gdata_one[:,t] = Gseries[stateargs]

    # Gfdata = {}
    # Gfdata_one = {}
    # for i in range(Nstates):
    #     Gfdata['%s_%d'%(key,i)] = np.copy(Gdata[:,i,:].real)
        # Gfdata_one['%s_%d'%(key,i)] = np.copy(Gdata_one[i,:].real)

    return Gdata, vec#, Gdata_one, Gfdata, Gfdata_one





meson_list = ["I0", "I1"]
irrp_list = ["P[000]","P[001]","P[002]","P[111]","P[011]"]
chdim = len(irrp_list)
c2pt11 = np.load("./data/00_00.npy")
Ncnfg = c2pt11.shape[0]
T = c2pt11.shape[1]
T_hlf = T//2

for k in range(len(meson_list)):
    c2pt = np.zeros((chdim,chdim,Ncnfg,T),dtype=complex)
    for i in range(chdim):
        for j in range(chdim):
            c2pt[i,j] = np.load("./data/%02d_%d%d.npy"%(k,i,j))

    c2pt_jcknf = np.zeros((chdim,chdim,Ncnfg,T),dtype=complex)
    for i in range(chdim):
        for j in range(chdim):
            c2pt_sum = np.sum(c2pt[i,j],axis=0)
            c2pt_jcknf[i,j] = ((c2pt_sum) - c2pt[i,j])/(Ncnfg-1)

    c2pt_jcknf = c2pt_jcknf.transpose(2,0,1,3)


    t0 = 5
    tref = 6
    lam = np.zeros((Ncnfg,chdim,T),dtype=complex)
    lam,vec = GEVP(T, c2pt_jcknf, t0, tref)
    lam = lam.transpose(1,0,2)
    lamr = np.zeros((chdim,Ncnfg,T))
    lamr = lam.real

    # 创建绘图
    fig, ax = plt.subplots(1, 1, figsize=(10, 7*0.5))
    # 为不同通道定义颜色和标记
    colors = ['red', 'orange', 'blue', 'green', 'purple']
    markers = ['o', 's', 'D', '^', 'x']
    # 设置最大时间点用于绘图
    t_max_plot = 30
    for idx in range(chdim):
        # 保存特征值
        np.save("./data/%02d_lam_jack_%d.npy"%(k,idx), lamr[idx])
        # 计算有效质量 (m_t = ln(C_t / C_{t+1}))
        mass = np.log(lamr[idx,:,0:-1] / lamr[idx,:,1:])
        mass = np.abs(mass)  # 取绝对值确保质量为正
        # 计算平均值和误差
        mass_mean = np.mean(mass, axis=0)
        mass_err = np.sqrt(Ncnfg - 1) * np.std(mass, axis=0)
        # 绘制带误差棒的数据，添加误差帽(capsize)和更明显的线条
        ax.errorbar(
            np.arange(1, t_max_plot),
            mass_mean[1:t_max_plot],
            yerr=mass_err[1:t_max_plot],
            fmt=markers[idx],
            color=colors[idx],
            capsize=4,          # 误差帽大小
            capthick=1.5,       # 误差帽线条粗细
            elinewidth=1.5,     # 误差棒线条粗细
            markersize=3,      # 标记大小（修正拼写）
            markeredgewidth=0.5,  # 标记边缘线条粗细（修正拼写）
            label=f'Channel {idx}'
        )
    # 添加垂直参考线和固定位置的文本
    ax.axvline(x=t0, linestyle='--', color='red', alpha=0.7)
    ax.text(t0, 1.6, '$t_0$', fontsize=12, color='red')  # 固定y位置为1.45
    # 设置坐标轴标签和范围
    plt.ylim(0.3, 2.0)
    plt.xlim(0, 31)
    plt.xlabel('t/a', fontsize=12)
    plt.ylabel('$aE_0$', fontsize=12)
    # 添加图例
    plt.legend()
    # 添加标题
    plt.title(f'Effective Mass for {meson_list[k]}', fontsize=14)
    # 调整布局并保存
    plt.tight_layout()
    fig.savefig("./output/%02d_lam_jack_orig.png"%(k))