import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
import plotparams
plotparams.buba()

# constants
y_0 = 1.e-5 
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
h = 0.6774
measurement = "simulation" # "aperture" 
if measurement == "simulation":
    N_gal = 50000
    #snapshot = 91; z = 0.1
    snapshot = 99; z = 0.0
    save_dir = "/mnt/marvin1/boryanah/SZ_TNG/"
    y500c = np.load(f"{save_dir}/Y_500c_nr_{N_gal:d}_snap_{snapshot:d}.npy")
    m500c = np.load(f"{save_dir}/M_500c_nr_{N_gal:d}_snap_{snapshot:d}.npy")/h
else:
    N_gal = 31000
    orientation = "xy"
    snapshot = 67 # 91 # 78 # 67
    z = 0.5
    aperture_mode = "r500"
    #aperture_mode = "fixed"
    if aperture_mode == "r500":
        data_fn = f"{save_dir}/galaxies_AP{aperture_mode}_top{N_gal:d}_{orientation}_{snapshot:d}_fp.npz"
    else:
        theta_arcmin = 2.1 # 1.3
        data_fn = f"{save_dir}/galaxies_th{theta_arcmin:.1f}_top{N_gal:d}_{orientation}_{snapshot:d}_fp.npz"
    data = np.load(data_fn)
    m500c = data['mstar']/h
    y500c = data['y_disks']


mmin = 1.e12
mmax = 1.e15
choice = (m500c < mmax) & (m500c > mmin)
print("number of clusters = ", np.sum(choice))
m500c = m500c[choice]
y500c = y500c[choice]

Om_m = 0.3083
Om_Lambda = 1 - Om_m
E_z = np.sqrt(Om_m*(1+z)**3 + Om_Lambda)
print("Ez-2/3", E_z**(-2./3))

x_sgns = m500c
y_sgns = y500c*E_z**(-2./3)*(500.)**2/ (180./np.pi)**2.

def get_sbpl_params(tau_sgns, A_sgns, std=None):
    #a1, a2, A, d, Xp = 2.4, 1.6, -3.6, 0.18, np.log10(5.2e13)
    a1, a2, A, d, Xp = 2.418, 1.687, -3.682, 0.25, np.log10(3.7e13)
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
    
def get_lin_fit(tau_sgns, A_sgns):
    A_0 = 1.e14 #Msun # can do this with: from functools import partial
    popt, pcov = curve_fit(func, np.log(A_sgns/A_0), np.log(tau_sgns))
    ln_tau_0, m = popt[0], popt[1]
    print("ln tau 0, m = ", ln_tau_0, m)
    #y_binc = np.geomspace(y_sgns.min(), y_sgns.max(), 20)
    tau_fit = np.exp(func(np.log(A_sgns/A_0), ln_tau_0, m)) #lin_mod(y_binc, y_0, ln_tau_0, m)
    sigma_ln_tau = np.sqrt(np.sum((np.log(tau_fit)-np.log(tau_sgns))**2)/(len(tau_sgns)-1))
    print("sigma_ln_tau = ", sigma_ln_tau)
    return tau_fit, A_sgns

logbins = np.linspace(12, 13.9, 20)
logbins = np.hstack((logbins, np.linspace(14, 14.6, 4)))
#logbins = np.linspace(12, 13.7, 18)
#logbins = np.hstack((logbins, np.linspace(13.8, 14.6, 4)))
#logbins = np.hstack((logbins, np.linspace(13.8, 14.7, 5)))
print(logbins)

logbinc = (logbins[1:]+logbins[:-1])*.5
bins = 10.**logbins
binc = 10.**logbinc
"""
mu, _, binno = binned_statistic(x_sgns, y_sgns, 'mean', bins=bins)
std, _, binno = binned_statistic(x_sgns, y_sgns, 'std', bins=bins)
yerr_lo = std/2.
yerr_hi = std/2.
"""
log_mu, _, binno = binned_statistic(x_sgns, y_sgns, geom_mean, bins=bins)
log_std, _, binno = binned_statistic(x_sgns, y_sgns, geom_std, bins=bins)
mu = 10.**log_mu
std = 0.5*(10.**(log_mu+log_std)-10.**(log_mu-log_std))
yerr_lo, yerr_hi = 10.**(log_mu)-10.**(log_mu-log_std), 10.**(log_mu+log_std)-10.**(log_mu)

# linear fit (binned stats)
y_lin_fit, x_lin_fit = get_lin_fit(mu, binc)
# sbpl fit (element-wise)
#y_fit = get_sbpl_fit(y_sgns, x_sgns)
# sbpl fit (binned stats)
y_sbpl_fit, x_sbpl_fit, A, a1, a2, d, Xp = get_sbpl_fit(mu, binc, std=std)

"""
# jackknife
n_jack = 50
inds = np.arange(len(x_sgns), dtype=int)
As, a1s, a2s, ds, Xps = np.zeros((5, n_jack))
np.random.shuffle(inds)
x_sgns, y_sgns = x_sgns[inds], y_sgns[inds]
for i in range(n_jack):
    sel = get_group(i, n_jack, len(x_sgns))
    log_mu, _, binno = binned_statistic(x_sgns[sel], y_sgns[sel], geom_mean, bins=bins)
    log_std, _, binno = binned_statistic(x_sgns[sel], y_sgns[sel], geom_std, bins=bins)
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
plt.figure(1, figsize=(9, 7))
plt.scatter(x_sgns, y_sgns, s=10, alpha=0.1, color=hexcolors_bright[0], marker='x')
plt.errorbar(binc, mu, yerr=[yerr_lo, yerr_hi], color=hexcolors_bright[3], capsize=4, label='simulation', zorder=1)
plt.plot(x_lin_fit, y_lin_fit, color=hexcolors_bright[1], label='linear', zorder=2)
plt.plot(x_sbpl_fit, y_sbpl_fit, color=hexcolors_bright[4], label='SBPL', zorder=3)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.ylim(0.9*y_sgns.min(), 1.1*y_sgns.max())
plt.xlim(0.9*x_sgns.min(), 1.1*x_sgns.max())
#plt.xlabel(r'$M_\ast \ [M_\odot/h]$')
plt.xlabel(r'$M_{500c} \ [M_\odot]$')
plt.ylabel(r'$Y_{\rm SZ, 500} \ [{\rm Mpc}^2]$')
plt.savefig("figs/m500c_y500c.png")

plt.figure(2, figsize=(9, 7))
plt.plot(binc, np.zeros_like(binc), 'k--')
plt.plot(binc, (mu-y_lin_fit)/std, color=hexcolors_bright[1], label='linear')
plt.plot(binc, (mu-y_sbpl_fit)/std, color=hexcolors_bright[4], label='SBPL')
plt.legend()
plt.xscale('log')
#plt.yscale('log')
plt.ylim([-2., 2.])
plt.xlim(0.9*x_sgns.min(), 1.1*x_sgns.max())
plt.xlabel(r'$M_{500c} \ [M_\odot]$')
plt.ylabel(r'$(Y_{\rm SZ, 500} - \hat Y_{\rm SZ, 500}) / \sigma$')
plt.savefig("figs/m500c_y500c_diff.png")
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
