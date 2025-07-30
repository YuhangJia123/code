import gvar as gv
import lsqfit
import numpy as np
import matplotlib.pyplot as plt

# c2pt_raw = np.load("./data_2pt.npy")
# c2pt = c2pt_raw[:, 0, :, -2]
# c2pt = np.load("rho.npy")
# c2pt = np.load("pion.npy")
# c2pt = np.load("kaon.npy")

def fit_fwrd(menson_num: int, lam_num: int, mode:str):

    c2pt = np.load("./data/%02d_lam_%s_%d.npy"%(menson_num, mode, lam_num))
    Ncnfg = c2pt.shape[0]
    T = c2pt.shape[1]
    T_hlf = T//2

    #? jack-knife resample 
    #! only used in plot , compare original data with fit
    c2pt_sum = np.sum(c2pt,axis=0)
    c2pt_jcknf = ((c2pt_sum) - c2pt)/(Ncnfg-1) # jack-knife resample
    c2pt_jcknf = c2pt
    c2pt_cntrl = np.mean(c2pt_jcknf,axis=0) # jack-knife mean
    c2pt_err = np.sqrt(Ncnfg-1)*np.std(c2pt_jcknf,axis=0) # jack-knife std

    #? drop the first data point, normalized to c2[1] ~ 1
    t_ary = np.array(range(1,T))
    c2pt_jcknf = c2pt_jcknf/c2pt_cntrl[1]
    c2pt_cntrl = np.mean(c2pt_jcknf,axis=0) # jack-knife mean
    c2pt_err = np.sqrt(Ncnfg-1)*np.std(c2pt_jcknf,axis=0) # jack-knife std

    #? average forward/backward propgation, for better symmetry 
    c2pt_jcknf_fwrd = c2pt_jcknf[:,1:T_hlf+1] #? forward propgation part
    c2pt_jcknf_bwrd = c2pt_jcknf[:,T_hlf:][:,::-1] #? backward propgation part
    #c2pt_jcknf_avg = (c2pt_jcknf_fwrd+c2pt_jcknf_bwrd)/2 #? average forward/backward propgation
    c2pt_fwrd_cntrl = np.mean(c2pt_jcknf_fwrd,axis=0) # jack-knife mean
    c2pt_fwrd_cov = (Ncnfg-1)*np.cov(np.transpose(c2pt_jcknf_fwrd,axes=(1,0))) # jack-knife covariance

    def ft_mdls(t_dctnry, p):
        mdls = {}
        ts = t_dctnry['c2pt']
        #mdls['c2pt'] = p['c0']*np.exp(-p['E0']*ts)
        mdls['c2pt'] = p['c0']*np.exp(-p['E0']*T_hlf)*np.cosh(p['E0']*(ts-T_hlf))
        # mdls['c2pt'] = p['c0']*np.exp(-np.sqrt(p['E0']**2+)*T_hlf)*np.cosh(p['E0']*(ts-T_hlf))
        # mdls['c2pt'] = (1.0 - p['c0'])*np.exp(-p['E0']*(ts-5)) + p['c0']*np.exp(-p['E1']*(ts-5))
        return mdls

    ini_prr = gv.gvar({'c0': '0(5)','E0': '0.6(0.5)'}) #? initial value
    # ini_prr = gv.gvar({'c0': '0(5)','E0': '0.4(0.5)','E1': '0.4(5)'}) #? initial value

    chi2 = np.zeros((T_hlf,T_hlf))
    # for T_strt in range(5,T_hlf-10):
    if lam_num==3:
        range_a=range(2,12)
    elif menson_num==1 and lam_num==4:
        range_a=range(2,12)
    else:
        range_a=range(2,15)

    for T_strt in range_a:
        for T_end in range(T_strt+3, range_a.stop+3):
    #        print(T_strt,T_end)
            tsep_dctnry = {'c2pt': t_ary[T_strt:T_end]}
            c2pt_dctnry = {'c2pt': gv.gvar(c2pt_fwrd_cntrl[T_strt:T_end], c2pt_fwrd_cov[T_strt:T_end, T_strt:T_end])}
            fit = lsqfit.nonlinear_fit(data=(tsep_dctnry, c2pt_dctnry), fcn=ft_mdls, prior=ini_prr, debug=True) #? excute the fit
            #筛选出Q值符合的结果
            if fit.Q < 0.3 or fit.Q > 0.7: #? Q值筛选
                continue
            else:
                chi2[T_strt,T_end] = fit.chi2/fit.dof
    #        print(chi2[T_strt,T_end])

    # 找到所有与最小chi2距离相等的候选点
    dist = np.abs(chi2 - 1.0)
    min_dist = np.min(dist)
    candidates = np.argwhere(dist == min_dist)
    # 按区间长度(Tend-Tstrt)排序候选点（从大到小）
    candidates = candidates[np.argsort(candidates[:,1] - candidates[:,0])[::-1]]
    # 选择区间长度最大的点
    best_idx = candidates[0]
    print(f"({best_idx[0]}, {best_idx[1]})")

    T_strt = best_idx[0] #? starting point of the fit
    T_end =  best_idx[1] #? ending point of the fit
    # T_strt = 15 #? starting point of the fit
    # T_end =  25 #? ending point of the fit
    tsep_dctnry = {'c2pt': t_ary[T_strt:T_end]}
    c2pt_dctnry = {'c2pt': gv.gvar(c2pt_fwrd_cntrl[T_strt:T_end], c2pt_fwrd_cov[T_strt:T_end, T_strt:T_end])}
    fit = lsqfit.nonlinear_fit(data=(tsep_dctnry, c2pt_dctnry), fcn=ft_mdls, prior=ini_prr, debug=True) #? excute the fit
    # 输出结果到文件（用户要求的修改）
    # 提取E0的值和误差
    E0_val = fit.p['E0'].mean
    E0_err = fit.p['E0'].sdev
    output_file = ("./output/fit_%s_output"%(mode))
    try:
        with open(output_file, 'a') as f:
            # 添加描述语句（用户要求）
            f.write(f"\n\n===== 同位旋{menson_num}，第{lam_num+1}个能级，E0={E0_val}，error={E0_err} =====\n")
            f.write(f"[{T_strt}, {T_end}]\n")
            f.write(fit.format(True) + "\n")
    except IOError as e:
        print(f"文件写入错误: {e}")


    #? time slices used in fit
    t_ary = fit.data[0]['c2pt']

    #? data to be fitted
    c2pt_fwrd_dat_gvar = fit.data[1]['c2pt'] 
    c2pt_fwrd_dat_cntrl = np.array([c2.mean for c2 in c2pt_fwrd_dat_gvar])
    c2pt_fwrd_dat_err = np.array([c2.sdev for c2 in c2pt_fwrd_dat_gvar])

    #? fitted function values
    t_lst = np.linspace(5, T-5, 60) 
    c2pt_fit_fcn_gvar = fit.fcn({'c2pt':t_lst}, fit.p)['c2pt']
    c2pt_fit_fcn_cntrl = np.array([c2.mean for c2 in c2pt_fit_fcn_gvar])
    c2pt_fit_fcn_err = np.array([c2.sdev for c2 in c2pt_fit_fcn_gvar])

    #? plots
    fig, ax = plt.subplots(1,1, figsize=(10, 7*0.5))
    ax.errorbar(np.array(range(0,T)),c2pt_cntrl,yerr=c2pt_err,fmt='bo',alpha=0.4,label="$C_2$") #? plot original data
    ax.errorbar(t_ary,c2pt_fwrd_dat_cntrl,yerr=c2pt_fwrd_dat_err,fmt='go',alpha=0.5,label="frwrd. $C_2$") #? plot forward/backward averaged data
    ax.plot(t_lst,c2pt_fit_fcn_cntrl,color="b",label="best fit") #? plot fitted function
    ax.fill_between(t_lst,c2pt_fit_fcn_cntrl-c2pt_fit_fcn_err,c2pt_fit_fcn_cntrl+c2pt_fit_fcn_err,alpha=0.3) #? plot fitted function errorband
    plt.yscale("log")
    plt.xlabel('t/a')
    plt.ylabel('$C_2$')
    ax.legend(loc='upper center', fontsize=10, frameon=True, fancybox=True, framealpha=0.8, borderpad=0.3, \
                ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=1.5)
    fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    # fig.savefig("rho_fit.png")
    # fig.savefig("pion_fit.png")
    # fig.savefig("kaon_fit.png")
    fig.savefig("./output/%02d_%sfit_%d.png"%(menson_num, mode, lam_num))

    return E0_val, E0_err, T_strt, T_end

if __name__ == "__main__":
    menson_num = 0  # 同位旋编号
    lam_num = 0     # 能级编号
    mode = 'jack'   # 模式，'jack' 或 'boot'
    try:
        E0, errorbar, Tstart, Tend = fit_fwrd(menson_num, lam_num, mode)
        print(f"拟合结果: E0 = {E0:.4f} ± {errorbar:.4f}, 区间: [{Tstart}, {Tend}]")
    except Exception as e:
        print(f"发生错误: {e}")