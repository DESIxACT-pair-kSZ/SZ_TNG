import time

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pixell import enmap, utils, enplot
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.coordinates import SkyCoord
from numba_2pcf.cf import numba_pairwise_vel#, numba_2pcf

from tools import extractStamp, calc_T_AP#, get_tzav_fast

# physics constants in cgs
m_p = 1.6726e-24 # g
X_H = 0.76 # unitless
sigma_T = 6.6524587158e-25 # cm^2
m_e = 9.10938356e-28 # g
c = 29979245800. # cm/s
kpc_to_cm = ((1.*u.kpc).to(u.cm)).value # cm
Mpc_to_cm = kpc_to_cm*1000. # 3.086e+24 cm
solar_mass = 1.989e33 # g
sigma_T_over_m_p = sigma_T / m_p
solar_mass_over_Mpc_to_cm_squared = solar_mass/Mpc_to_cm**2

def get_jack_corr(xyz_true, w_true, Lbox, bins, N_dim=3, nthreads=16):
    
    # bins for the correlation function
    N_bin = len(bins)
    bin_centers = (bins[:-1] + bins[1:])/2.
    true_max = xyz_true.max()
    true_min = xyz_true.min()
    
    if true_max > Lbox or true_min < 0.:
        print("NOTE: we are wrapping positions")
        xyz_true = xyz_true % Lbox

    # empty arrays to record data
    Corr_true = np.zeros((N_bin-1, N_dim**3))
    for i_x in range(N_dim):
        for i_y in range(N_dim):
            for i_z in range(N_dim):
                print(i_x, i_y, i_z)
                xyz_true_jack = xyz_true.copy()
                w_true_jack = w_true.copy()
                
                xyz = np.array([i_x,i_y,i_z],dtype=int)
                size = Lbox/N_dim

                bool_arr = np.prod((xyz == (xyz_true/size).astype(int)),axis=1).astype(bool)
                xyz_true_jack[bool_arr] = np.array([0.,0.,0.])
                xyz_true_jack = xyz_true_jack[np.sum(xyz_true_jack,axis=1)!=0.]
                w_true_jack[bool_arr] = -1
                w_true_jack = w_true_jack[np.abs(w_true_jack+1) > 1.e-6]

                # in case we don't have weights
                xi_true = numba_pairwise_vel(xyz_true_jack, w_true_jack, box=Lbox, Rmax=np.max(rbins), nbin=len(rbins)-1, corrfunc=False, nthread=nthread, periodic=False)['pairwise']
                Corr_true[:,i_x+N_dim*i_y+N_dim**2*i_z] = xi_true
    
    # compute mean and error
    Corr_mean_true = np.mean(Corr_true, axis=1)
    Corr_err_true = np.sqrt(N_dim**3-1)*np.std(Corr_true, axis=1)

    return Corr_mean_true, Corr_err_true, bin_centers


# load galaxy information
N_gal = 30000
sigma_z = 0.01
Lbox = 205. # Mpc/h
orientation = "xy"
snapshot = 67 # 91 # 78 # 67
redshift = 0.5
aperture_mode = "r500"
#aperture_mode = "fixed"
theta_arcmin = 2.1 # 1.3
if aperture_mode == "r500":
    data_fn = f"data/galaxies_AP{aperture_mode}_top{N_gal:d}_{orientation}_{snapshot:d}_fp.npz"
else:
    data_fn = f"data/galaxies_th{theta_arcmin:.1f}_top{N_gal:d}_{orientation}_{snapshot:d}_fp.npz"
data = np.load(data_fn)
h = 0.6774
#mstar = data['mstar']/h
ra = data['RA']
dec = data['DEC']
mstar = np.load('mstar.npy')/h # Msun
mhalo = np.load('mhalo.npy')/h # Msun
pos = data['pos']/1000. # cMpc/h

# set up cosmology
h = 0.6774
cosmo = FlatLambdaCDM(H0=h*100, Om0=0.3089, Tcmb0=2.725)
H_z = cosmo.H(redshift).value
print("H(z) = ", H_z)


#vel = data['vel'] # km/s
# TESTING
vel = np.load('vel.npy') # km/s
pos[:, 2] += vel[:, 2]*(1.+redshift)/H_z*h
tau_disk_mean = data['tau_disks']
y_disk_mean = data['y_disks']
b_disk_mean = data['b_disks']
tau_annulus_mean = data['tau_rings']
y_annulus_mean = data['y_rings']
b_annulus_mean = data['b_rings']
divs = np.ones(len(ra))
#disk_std = data['disk_stds'] 
#annulus_std = data['ring_stds']

mmin = 1.e12
mmax = 1.e15
choice = (mhalo < mmax) & (mhalo > mmin)
print("number of clusters = ", np.sum(choice))
mstar = mstar[choice]
mhalo = mhalo[choice]
ra = ra[choice]
dec = dec[choice]
pos = pos[choice]
tau_disk_mean = tau_disk_mean[choice]
tau_annulus_mean = tau_annulus_mean[choice]
y_disk_mean = y_disk_mean[choice]
y_annulus_mean = y_annulus_mean[choice]
b_disk_mean = b_disk_mean[choice]
b_annulus_mean = b_annulus_mean[choice]
divs = divs[choice]

# define signals
#tau_sgns = np.subtract(tau_disk_mean, tau_annulus_mean)
#y_sgns = np.subtract(y_disk_mean, y_annulus_mean)
#tau_sgns = (tau_disk_mean)
#y_sgns = (y_disk_mean)
T_APs = b_disk_mean

# get the redshift-weighted apertures and temperature decrement around each galaxy
#bar_T_APs = get_tzav_fast(T_APs, Z, sigma_z)
#delta_Ts = T_APs - bar_T_APs
delta_Ts = T_APs

# define bins in Mpc
rbins = np.linspace(0., 80., 11) # Mpc/h
rbinc = (rbins[1:]+rbins[:-1])*.5 # Mpc/h
nthread = 1 #os.cpu_count()//4

def get_bias(Mvirs, redshifts):
    """ get bias as a function of halo mass (units Modot/h) and redshift """
    #nu = peaks.peakHeight(M, z)
    #b = bias.haloBiasFromNu(nu, model = 'sheth01')
    b = bias.haloBias(Mvirs, model='tinker10', z=redshifts, mdef='200m')#mdef='vir')
    return b

def get_tau_theory(M_star, M_vir, d_A=1.):
    # M_vir within 2.1 arcmin in units of Modot; d_A in Mpc
    f_b = 0.157 # Omega_b/Omega_m
    f_star = M_star/M_vir
    x_e = (X_H + 1.)/(2.*X_H)
    tau = sigma_T_over_m_p * solar_mass_over_Mpc_to_cm_squared * x_e * X_H * (1-f_star) * f_b * M_vir / d_A**2
    return tau

bias = get_bias(mhalo, redshift)
tau = get_tau_theory(mstar, mhalo)

print(pos[:, 0].min(), pos[:, 1].min(), pos[:, 2].min(), pos[:, 0].max(), pos[:, 1].max(), pos[:, 2].max(), delta_Ts.min(), delta_Ts.max(), delta_Ts.mean())
t = time.time()
#delta_Ts = np.ones_like(delta_Ts)
#pos = np.random.rand(pos.shape[0], pos.shape[1])*205.
table = numba_pairwise_vel(pos, delta_Ts, box=None, Rmax=np.max(rbins), nbin=len(rbins)-1, corrfunc=False, nthread=nthread, periodic=False)
DD = table['npairs']
PV = table['pairwise']
print("calculation took = ", time.time()-t)

want_error = 1
if want_error:
    PV_mean, PV_err, _ = get_jack_corr(pos, delta_Ts, Lbox, rbins, N_dim=3, nthreads=nthread)
    print("chi2, dof = ", np.sum((PV/PV_err)**2.), len(PV_mean))
    plt.plot(rbinc, np.zeros(len(rbinc)), ls='--')
    plt.errorbar(rbinc, -PV, yerr=PV_err)
    plt.errorbar(rbinc, -PV_mean, yerr=PV_err)
    plt.show()
else:
    plt.plot(rbinc, np.zeros(len(rbinc)), ls='--')
    plt.plot(rbinc, -PV)
    plt.show()
