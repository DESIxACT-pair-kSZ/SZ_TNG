import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
import plotparams
plotparams.buba()

# constants
y_0 = 1.e-5 #y_sgns.min() #1.e-6
Xn = 1.e14 #Msun # can do this with: from functools import partial

# colors
hexcolors_bright = ['#CC3311','#0077BB','#EE7733','limegreen','#BBBBBB','#33BBEE','#EE3377','#0099BB']
greysafecols = ['#809BC8', 'black', '#FF6666', '#FFCC66', '#64C204']

def Y_SBPL(X, A, a1, a2, d, Xp):
    #Yf = 10.**A*(X/Xn)**(0.5*(a2+a1))*(np.cosh(np.log10(X/Xp)/d)/np.cosh(np.log10(Xn/Xp)/d))**(0.5*(a2-a1)*d*np.log(10.))
    Yf = 10.**A*(X/Xn)**(0.5*(a2+a1))*(np.cosh(np.log10(X/10.**Xp)/d)/np.cosh(np.log10(Xn/10.**Xp)/d))**(0.5*(a2-a1)*d*np.log(10.))
    return Yf

def geom_mean(x):
    n = len(x)
    log_mu = (np.sum(np.log10(x))/n)
    return log_mu

def geom_std(x):
    n = len(x)
    log_mu = geom_mean(x)
    log_std = (np.sqrt(np.sum((np.log10(x)-log_mu)**2)/n))
    return log_std

def func(x, a, b):
    return a + b * x

def lin_mod(y, y_0, ln_tau_0, m):
    return ln_tau_0 + m * np.log(y/y_0)

def get_group(i, N_groups, N_all):
    len_Group = N_all//N_groups
    sel = np.ones(N_all, dtype=bool)
    #if i == 0: # I don't think this makes sense?
    #sel[0] = False
    #else:
    #sel[i * len_Group:(i+1) * len_Group] = False 
    sel[i * len_Group : (i+1) * len_Group] = False
    return sel

# load galaxy information
N_gal = 31000
orientation = "xy"
snapshot = 67 # 91 # 78 # 67
aperture_mode = "r500"
#aperture_mode = "fixed"
theta_arcmin = 2.1 # 1.3
if aperture_mode == "r500":
    data_fn = f"data/galaxies_AP{aperture_mode}_top{N_gal:d}_{orientation}_{snapshot:d}_fp.npz"
else:
    data_fn = f"data/galaxies_th{theta_arcmin:.1f}_top{N_gal:d}_{orientation}_{snapshot:d}_fp.npz"
data = np.load(data_fn)
h = 0.6774
mstar = data['mstar']/h
ra = data['RA']
dec = data['DEC']
lum = data['mstar']
tau_disk_mean = data['tau_disks']
y_disk_mean = data['y_disks']
tau_annulus_mean = data['tau_rings']
y_annulus_mean = data['y_rings']
divs = np.ones(len(ra))
#disk_std = data['disk_stds'] 
#annulus_std = data['ring_stds']

mmin = 1.e12
mmax = 1.e15
choice = (mstar < mmax) & (mstar > mmin)
print("number of clusters = ", np.sum(choice))
mstar = mstar[choice]
ra = ra[choice]
dec = dec[choice]
lum = lum[choice]
tau_disk_mean = tau_disk_mean[choice]
tau_annulus_mean = tau_annulus_mean[choice]
y_disk_mean = y_disk_mean[choice]
y_annulus_mean = y_annulus_mean[choice]
divs = divs[choice]

# define signals
#tau_sgns = np.subtract(tau_disk_mean, tau_annulus_mean)
#y_sgns = np.subtract(y_disk_mean, y_annulus_mean)
tau_sgns = (tau_disk_mean)
y_sgns = (y_disk_mean)

def get_sbpl_params(tau_sgns, A_sgns, std=None):
    a1, a2, A, d, Xp = 2.4, 1.6, -3.6, 0.18, np.log10(5.2e13)
    print(f"init: A, a1, a2, d, Xp = {A:.2f}, {a1:.2f}, {a2:.2f}, {d:.2f}, {Xp:.2f}")
    p0 = [A, a1, a2, d, Xp]
    popt, pcov = curve_fit(Y_SBPL, A_sgns, tau_sgns, p0=p0, bounds=((-8., 0., 0., 0.01, 12), (-1., 6., 6., 0.5, 16)), sigma=std)#, method='dogbox')
    A, a1, a2, d, Xp = popt[0], popt[1], popt[2], popt[3],popt[4]
    print(f"mine: A, a1, a2, d, Xp = {A:.2f}, {a1:.2f}, {a2:.2f}, {d:.2f}, {Xp:.2f}")
    return A, a1, a2, d, Xp

def get_sbpl_fit(tau_sgns, A_sgns, std=None):
    A, a1, a2, d, Xp = get_sbpl_params(tau_sgns, A_sgns, std=std)
    tau_fit = Y_SBPL(A_sgns, A, a1, a2, d, Xp)
    sigma_ln_tau = np.sqrt(np.sum((np.log(tau_fit)-np.log(tau_sgns))**2)/(len(tau_sgns)-1))
    print("sigma_ln_tau = ", sigma_ln_tau)
    return tau_fit, A_sgns, A, a1, a2, d, Xp
    
def get_lin_fit(tau_sgns, A_sgns, A_0):
    popt, pcov = curve_fit(func, np.log(A_sgns/A_0), np.log(tau_sgns))
    ln_tau_0, m = popt[0], popt[1]
    print("ln tau 0, m = ", ln_tau_0, m)
    #y_binc = np.geomspace(y_sgns.min(), y_sgns.max(), 20)
    tau_fit = np.exp(func(np.log(A_sgns/A_0), ln_tau_0, m)) #lin_mod(y_binc, y_0, ln_tau_0, m)
    sigma_ln_tau = np.sqrt(np.sum((np.log(tau_fit)-np.log(tau_sgns))**2)/(len(tau_sgns)-1))
    print("sigma_ln_tau = ", sigma_ln_tau)
    return tau_fit

"""
# tau-Y fit 
tau_fit = get_lin_fit(tau_sgns, y_sgns, y_0)

plt.figure(figsize=(9, 7))
plt.scatter(y_sgns, tau_sgns, s=10, alpha=0.1, color=hexcolors_bright[0], marker='x')
plt.plot(y_sgns, tau_fit)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.9*y_sgns.min(), 1.1*y_sgns.max())
plt.ylim(0.9*tau_sgns.min(), 1.1*tau_sgns.max())
plt.xlabel(r'$\langle y \rangle_\Theta$')
plt.ylabel(r'$\langle \tau \rangle_\Theta$')
plt.savefig("figs/y_tau.png")

# tau-M500 fit 
tau_fit = get_lin_fit(tau_sgns, mstar, Xn)

plt.figure(figsize=(9, 7))
plt.scatter(mstar, tau_sgns, s=10, alpha=0.1, color=hexcolors_bright[0], marker='x')
plt.plot(mstar, tau_fit)
plt.xscale('log')
plt.yscale('log')
plt.ylim(0.9*tau_sgns.min(), 1.1*tau_sgns.max())
plt.xlim(0.9*mstar.min(), 1.1*mstar.max())
#plt.xlabel(r'$M_\ast \ [M_\odot/h]$')
plt.xlabel(r'$M_{500c} \ [M_\odot]$')
plt.ylabel(r'$\langle \tau \rangle_\Theta$')
plt.savefig("figs/mstar_tau.png")
plt.show()
"""

#bins = np.logspace(12, 14.6, 23)
logbins = np.linspace(12, 14.6, 23)
logbinc = (logbins[1:]+logbins[:-1])*.5
bins = 10.**logbins
binc = 10.**logbinc
#mu, _, binno = binned_statistic(mstar, y_sgns, 'mean', bins=bins)
#std, _, binno = binned_statistic(mstar, y_sgns, 'std', bins=bins)
log_mu, _, binno = binned_statistic(mstar, y_sgns, geom_mean, bins=bins)
log_std, _, binno = binned_statistic(mstar, y_sgns, geom_std, bins=bins)
mu = 10.**log_mu
std = 0.5*(10.**(log_mu+log_std)-10.**(log_mu-log_std))

# linear fit (element-wise)
#y_fit = get_lin_fit(y_sgns, mstar, Xn)
# sbpl fit (element-wise)
#y_fit = get_sbpl_fit(y_sgns, mstar)
# sbpl fit (binned stats)
y_fit, x_fit, A, a1, a2, d, Xp = get_sbpl_fit(mu, binc, std=std)

"""
# jackknife
n_jack = 50
inds = np.arange(len(mstar), dtype=int)
As, a1s, a2s, ds, Xps = np.zeros((5, n_jack))
np.random.shuffle(inds)
mstar, y_sgns = mstar[inds], y_sgns[inds]
for i in range(n_jack):
    sel = get_group(i, n_jack, len(mstar))
    log_mu, _, binno = binned_statistic(mstar[sel], y_sgns[sel], geom_mean, bins=bins)
    log_std, _, binno = binned_statistic(mstar[sel], y_sgns[sel], geom_std, bins=bins)
    mu = 10.**log_mu
    std = 0.5*(10.**(log_mu+log_std)-10.**(log_mu-log_std))
    As[i], a1s[i], a2s[i], ds[i], Xps[i] = get_sbpl_params(mu, binc, std=std)

def get_jack_mean_std(x):
    mean_x = np.mean(x)
    std_x = np.sqrt(((n_jack-1.)/n_jack)*np.sum((x-mean_x)**2.0))
    return mean_x, std_x

mean_A, std_A = get_jack_mean_std(As)
mean_a1, std_a1 = get_jack_mean_std(a1s)
mean_a2, std_a2 = get_jack_mean_std(a2s)
mean_d, std_d = get_jack_mean_std(ds)
mean_Xp, std_Xp = get_jack_mean_std(Xps)
print("A: all, mean, std = ", A, mean_A, std_A)
print("a1: all, mean, std = ", a1, mean_a1, std_a1)
print("a2: all, mean, std = ", a2, mean_a2, std_a2)
print("d: all, mean, std = ", d, mean_d, std_d)
print("Xp: all, mean, std = ", Xp, mean_Xp, std_Xp)
"""

# plot scatter, fit and binned
plt.figure(figsize=(9, 7))
plt.scatter(mstar, y_sgns, s=10, alpha=0.1, color=hexcolors_bright[0], marker='x')
plt.errorbar(binc, mu, yerr=[10.**(log_mu)-10.**(log_mu-log_std), 10.**(log_mu+log_std)-10.**(log_mu)], color=hexcolors_bright[3], capsize=4)
plt.plot(x_fit, y_fit)
plt.xscale('log')
plt.yscale('log')
plt.ylim(0.9*y_sgns.min(), 1.1*y_sgns.max())
plt.xlim(0.9*mstar.min(), 1.1*mstar.max())
#plt.xlabel(r'$M_\ast \ [M_\odot/h]$')
plt.xlabel(r'$M_{500c} \ [M_\odot]$')
plt.ylabel(r'$\langle y \rangle_\Theta$')
plt.savefig("figs/mstar_y.png")
plt.show()
quit()

# number of bins
n_bins = 8
mstar_thresh = np.zeros(n_bins+1)
N_gal_bin = N_gal//n_bins
jump = 0
for i in range(n_bins):
    mstar_thresh[i] = mstar[jump]
    jump += N_gal_bin
mstar_thresh[-1] = mstar[-1]

# number of jackknifes
N_jack = 50

tau_avg_bin = np.zeros(n_bins)
y_avg_bin = np.zeros(n_bins)
tau_err_bin = np.zeros(n_bins)
y_err_bin = np.zeros(n_bins)
# for each bin, coadd everyone
for i in range(n_bins):
    choice = (mstar_thresh[i+1] < mstar) & (mstar <= mstar_thresh[i])
    print("choice = ", np.sum(choice))
    dt = tau_sgns[choice]
    dy = y_sgns[choice]
    divsbin = divs[choice]

    # compute average signal in bin
    tau_avg_bin[i] = np.sum(np.multiply(dt, divsbin))/np.sum(divsbin)
    y_avg_bin[i] = np.sum(np.multiply(dy, divsbin))/np.sum(divsbin)
    tau_avg_jack = np.zeros(N_jack)
    y_avg_jack = np.zeros(N_jack)
    for j in range(N_jack):
        sel = get_group(j, N_jack, len(dt))
        tau_avg_jack[j] = np.sum(np.multiply(dt[sel], divsbin[sel]))/np.sum(divsbin[sel])
        y_avg_jack[j] = np.sum(np.multiply(dy[sel], divsbin[sel]))/np.sum(divsbin[sel])
    tau_err_bin[i] = np.sqrt(((N_jack-1.)/N_jack)*np.sum((tau_avg_jack-tau_avg_bin[i])**2.0))
    y_err_bin[i] = np.sqrt(((N_jack-1.)/N_jack)*np.sum((y_avg_jack-y_avg_bin[i])**2.0))
    print("bin information:")
    print(f"stellar mass threshold = {mstar_thresh[i+1]:.2e}")
    print(f"average tau = {tau_avg_bin[i]:.3e}")
    print(f"error tau = {tau_err_bin[i]:.3e}")
    print(f"average y = {y_avg_bin[i]:.3e}")
    print(f"error y = {y_err_bin[i]:.3e}")
    print("--------------------------------")

plt.figure(figsize=(9, 7))
plt.scatter(y_sgns, tau_sgns, s=10, alpha=0.1, color=hexcolors_bright[0], marker='x')
plt.errorbar(y_avg_bin, tau_avg_bin, xerr=y_err_bin, yerr=tau_err_bin, color=hexcolors_bright[1], marker='o', ls='', capsize=4)
plt.xscale('log')
plt.yscale('log')
#plt.xlim(0.9*y_avg_bin.min(), 1.1*y_avg_bin.max())
plt.xlim(0.9*y_sgns.min(), 1.1*y_sgns.max())
#plt.ylim(0.9*tau_avg_bin.min(), 1.1*tau_avg_bin.max())
plt.ylim(0.9*tau_sgns.min(), 1.1*tau_sgns.max())
plt.xlabel(r'$\langle y \rangle_\Theta$')
plt.ylabel(r'$\langle \tau \rangle_\Theta$')
plt.savefig("figs/y_tau.png")
#plt.show()

plt.figure(figsize=(9, 7))
plt.errorbar(mstar_thresh[1:], tau_avg_bin, yerr=tau_err_bin, color=hexcolors_bright[1], marker='o', ls='', capsize=4)
plt.xscale('log')
plt.yscale('log')
plt.ylim(0.9*tau_avg_bin.min(), 1.1*tau_avg_bin.max())
#plt.xlim(0.9*mstar_thresh.min(), 1.1*mstar_thresh.max())
plt.xlabel(r'$M_\ast \ [M_\odot/h]$')
plt.ylabel(r'$\langle \tau \rangle_\Theta$')
plt.savefig("figs/mstar_tau.png")
plt.show()
