import numpy as np
import matplotlib.pyplot as plt
import plotparams
plotparams.buba()

# colors
hexcolors_bright = ['#CC3311','#0077BB','#EE7733','limegreen','#BBBBBB','#33BBEE','#EE3377','#0099BB']
greysafecols = ['#809BC8', 'black', '#FF6666', '#FFCC66', '#64C204']

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
N_gal = 10000
orientation = "xy"
snapshot = 67
data = np.load(f"data/galaxies_top{N_gal:d}_{orientation}_{snapshot:d}_fp.npz")
mstar = data['mstar']
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

# define signals
#tau_sgns = np.subtract(tau_disk_mean, tau_annulus_mean)
#y_sgns = np.subtract(y_disk_mean, y_annulus_mean)
tau_sgns = (tau_disk_mean)
y_sgns = (y_disk_mean)

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
