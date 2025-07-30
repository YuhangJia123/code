import numpy as np
# from scipy.linalg import eigvalsh,eigvals
from scipy import linalg
import matplotlib.pyplot as plt

chdim = 4
c2pt11 = np.load("00.npy")
Ncnfg = c2pt11.shape[0]
T = c2pt11.shape[1]
T_hlf = T//2

c2pt = np.zeros((chdim,chdim,Ncnfg,T),dtype=complex)
for i in range(chdim):
    for j in range(chdim):
        c2pt[i,j] = np.load("%d%d.npy"%(i,j))

c2pt_jcknf = np.zeros((chdim,chdim,Ncnfg,T),dtype=complex)
for i in range(chdim):
    for j in range(chdim):
        c2pt_sum = np.sum(c2pt[i,j],axis=0)
        c2pt_jcknf[i,j] = ((c2pt_sum) - c2pt[i,j])/(Ncnfg-1)

c2pt_jcknf = c2pt_jcknf.transpose(2,0,1,3)
# def GEVP(params, corrmat, corrmat_one, t0, tref, key):
def GEVP(T, corrmat, t0, tref):
    # corrmat = corrmat.real
    # corrmat_one = corrmat_one.real
    Nstates = corrmat.shape[1]
    relen = corrmat.shape[0]
    Gdata = np.zeros([relen,Nstates,T],dtype=complex)
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
            Gdata[i,:,t] = Gseries[stateargs]

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

    return Gdata#, Gdata_one, Gfdata, Gfdata_one

t0 = 5
tref = 6
lam = np.zeros((Ncnfg,chdim,T),dtype=complex)
lam = GEVP(T, c2pt_jcknf, t0, tref)
lam = lam.transpose(1,0,2)

lamr = np.zeros((chdim,Ncnfg,T))
lamr = lam.real

fig, ax = plt.subplots(1,1, figsize=(10, 7*0.5))
for idx in range(chdim):
    np.save("lam%d.npy"%(idx),lamr[idx])
    # mass = np.log(lamr[idx,:,t0:-1]/lamr[idx,:,t0+1:])
    mass = np.log(lamr[idx,:,0:-1]/lamr[idx,:,0+1:])
    mass = np.where(mass>0,mass,-mass)
    mass_mean = np.mean(mass,axis=0)
    mass_err = np.sqrt(Ncnfg-1)*np.std(mass,axis=0)

    # fig, ax = plt.subplots(1,1, figsize=(10, 7*0.5))
    # ax.plot(np.array(range(1,T-1)),np.log(c2pt_cntrl[1:-1]/c2pt_cntrl[2:]),'bo')
    # ax.errorbar(np.array(range(1,T-t0)),mass_mean,yerr=mass_err,fmt='bo',alpha=0.4)
    ax.errorbar(np.array(range(1,30)),mass_mean[1:30],yerr=mass_err[1:30],fmt='o')
    # plt.xlabel('t/a')
    # plt.ylabel('$aE_0$')
    # fig.savefig("lam%d_mass.png"%(idx))

plt.ylim((0.3,1.5))
plt.xlabel('t/a')
plt.ylabel('$aE_0$')
ax.axvline(x=t0, linestyle='--')
ax.text(t0, ax.get_ylim()[1], '$t_0$', verticalalignment='bottom')
fig.savefig("lam_vec.png")