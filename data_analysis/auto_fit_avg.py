import gvar as gv
import lsqfit
import numpy as np
import matplotlib.pyplot as plt

# c2pt_raw = np.load("./data_2pt.npy")
# c2pt = c2pt_raw[:, 0, :, -2]
# c2pt = np.load("rho.npy")
# c2pt = np.load("pion.npy")
# c2pt = np.load("kaon.npy")

menson_num = 0
lam_num = 0

c2pt = np.load("%02d_lam%d.npy"%(menson_num,lam_num))
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
c2pt_jcknf_avg = (c2pt_jcknf_fwrd+c2pt_jcknf_bwrd)/2 #? average forward/backward propgation
c2pt_avg_cntrl = np.mean(c2pt_jcknf_avg,axis=0) # jack-knife mean
c2pt_avg_cov = (Ncnfg-1)*np.cov(np.transpose(c2pt_jcknf_avg,axes=(1,0))) # jack-knife covariance

def ft_mdls(t_dctnry, p):
    mdls = {}
    ts = t_dctnry['c2pt']
    mdls['c2pt'] = p['c0']*np.exp(-p['E0']*T_hlf)*np.cosh(p['E0']*(ts-T_hlf))
    # mdls['c2pt'] = p['c0']*np.exp(-np.sqrt(p['E0']**2+)*T_hlf)*np.cosh(p['E0']*(ts-T_hlf))
    # mdls['c2pt'] = (1.0 - p['c0'])*np.exp(-p['E0']*(ts-5)) + p['c0']*np.exp(-p['E1']*(ts-5))
    return mdls

ini_prr = gv.gvar({'c0': '0(5)','E0': '0.4(0.5)'}) #? initial value
# ini_prr = gv.gvar({'c0': '0(5)','E0': '0.4(0.5)','E1': '0.4(5)'}) #? initial value

chi2 = np.zeros((T,T))
# for T_strt in range(5,T_hlf-10):
for T_strt in range(5,25):
    for T_end in range(T_strt+2,25):
#        print(T_strt,T_end)
        tsep_dctnry = {'c2pt': t_ary[T_strt:T_end]}
        c2pt_dctnry = {'c2pt': gv.gvar(c2pt_avg_cntrl[T_strt:T_end], c2pt_avg_cov[T_strt:T_end, T_strt:T_end])}
        fit = lsqfit.nonlinear_fit(data=(tsep_dctnry, c2pt_dctnry), fcn=ft_mdls, prior=ini_prr, debug=True) #? excute the fit
        chi2[T_strt,T_end] = fit.chi2/fit.dof
#        print(chi2[T_strt,T_end])

idx = np.unravel_index(np.argmin(np.abs(chi2-1.0)),chi2.shape)
print(idx)

T_strt = idx[0] #? starting point of the fit
T_end =  idx[1] #? ending point of the fit
# T_strt = 15 #? starting point of the fit
# T_end =  25 #? ending point of the fit
tsep_dctnry = {'c2pt': t_ary[T_strt:T_end]}
c2pt_dctnry = {'c2pt': gv.gvar(c2pt_avg_cntrl[T_strt:T_end], c2pt_avg_cov[T_strt:T_end, T_strt:T_end])}
fit = lsqfit.nonlinear_fit(data=(tsep_dctnry, c2pt_dctnry), fcn=ft_mdls, prior=ini_prr, debug=True) #? excute the fit
print(fit.format(True)) #? print out fit results


#? time slices used in fit
t_ary = fit.data[0]['c2pt']

#? data to be fitted
c2pt_avg_dat_gvar = fit.data[1]['c2pt'] 
c2pt_avg_dat_cntrl = np.array([c2.mean for c2 in c2pt_avg_dat_gvar])
c2pt_avg_dat_err = np.array([c2.sdev for c2 in c2pt_avg_dat_gvar])

#? fitted function values
t_lst = np.linspace(10, T-10, 50) 
c2pt_fit_fcn_gvar = fit.fcn({'c2pt':t_lst}, fit.p)['c2pt']
c2pt_fit_fcn_cntrl = np.array([c2.mean for c2 in c2pt_fit_fcn_gvar])
c2pt_fit_fcn_err = np.array([c2.sdev for c2 in c2pt_fit_fcn_gvar])

#? plots
fig, ax = plt.subplots(1,1, figsize=(10, 7*0.5))
ax.errorbar(np.array(range(0,T)),c2pt_cntrl,yerr=c2pt_err,fmt='bo',alpha=0.4,label="$C_2$") #? plot original data
ax.errorbar(t_ary,c2pt_avg_dat_cntrl,yerr=c2pt_avg_dat_err,fmt='go',alpha=0.5,label="frwrd/bckwrd avg. $C_2$") #? plot forward/backward averaged data
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
fig.savefig("%02d_lam%d_fit.png"%(menson_num,lam_num))