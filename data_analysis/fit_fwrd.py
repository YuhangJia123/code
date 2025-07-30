import gvar as gv
import lsqfit
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    meson = 0 #? 0 for I0, 1 for I1
    chdim = 0 #? 0 for P[000], 1 for P[001], 2 for P[011], 3 for P[111], 4 for P[002]
    # c2pt_raw = np.load("./data_2pt.npy")
    # c2pt = c2pt_raw[:, 0, :, -2]
    c2pt = np.load("%02d_lam%d.npy"%(meson,chdim)) #? load the data, meson: 0 or 1, chdim: 0,1,2,3,4
    # c2pt = np.load("lam1.npy")
    Ncnfg = c2pt.shape[0]
    T = c2pt.shape[1]
    T_hlf = T//2

    #? jack-knife resample 
    #! only used in plot , compare original data with fit
    c2pt_sum = np.sum(c2pt,axis=0)
    c2pt_jcknf = ((c2pt_sum) - c2pt)/(Ncnfg-1) # jack-knife resample
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
    c2pt_jcknf_avg = (c2pt_jcknf_fwrd+c2pt_jcknf_bwrd)/2 #? average forward/backward propgation
    c2pt_fwrd_cntrl = np.mean(c2pt_jcknf_fwrd,axis=0) # jack-knife mean
    c2pt_fwrd_cov = (Ncnfg-1)*np.cov(np.transpose(c2pt_jcknf_fwrd,axes=(1,0))) # jack-knife covariance

    T_strt = 14 #? starting point of the fit
    T_end =  16 #? ending point of the fit
    tsep_dctnry = {'c2pt': t_ary[T_strt:T_end]}
    c2pt_dctnry = {'c2pt': gv.gvar(c2pt_fwrd_cntrl[T_strt:T_end], c2pt_fwrd_cov[T_strt:T_end, T_strt:T_end])}

    def ft_mdls(t_dctnry, p):
        mdls = {}
        ts = t_dctnry['c2pt']
        mdls['c2pt'] = p['c0']*np.exp(-p['E0']*T_hlf/2)*np.cosh(p['E0']*(ts-T_hlf))
        return mdls

    ini_prr = gv.gvar({'c0': '0(5)','E0': '0.2(0.5)'}) #? initial value
    fit = lsqfit.nonlinear_fit(data=(tsep_dctnry, c2pt_dctnry), fcn=ft_mdls, prior=ini_prr, debug=True) #? excute the fit
    print(fit.format(True)) #? print out fit results

    #? time slices used in fit
    t_ary = fit.data[0]['c2pt']

    #? data to be fitted
    c2pt_fwrd_dat_gvar = fit.data[1]['c2pt'] 
    c2pt_fwrd_dat_cntrl = np.array([c2.mean for c2 in c2pt_fwrd_dat_gvar])
    c2pt_fwrd_dat_err = np.array([c2.sdev for c2 in c2pt_fwrd_dat_gvar])

    #? fitted function values
    t_lst = np.linspace(10, T-10, 50) 
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
    fig.savefig("00_lam0_noauto_fit.png")
    # fig.savefig("lam1_fit.png")