"""
Script for obtaining beamed SZ maps
Size of box in degrees: (((1/1+z=1)*Lboxmpch)^2/d_Ampch^2*(180/pi)^2 
"""
import os
import gc
import sys

import numpy as np
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

"""
python3 apply_beam.py 2.1 0 TNG300; python3 apply_beam.py 1.3 0 TNG300; python3 apply_beam.py 2.1 1 TNG300; python3 apply_beam.py 1.3 1 TNG300;

python3 apply_beam.py 2.1 0 Illustris; python3 apply_beam.py 1.3 0 Illustris; python3 apply_beam.py 2.1 1 Illustris; python3 apply_beam.py 1.3 1 Illustris;

python3 apply_beam.py 1.6 0 TNG300; python3 apply_beam.py 1.6 1 TNG300; python3 apply_beam.py 1.6 0 Illustris; python3 apply_beam.py 1.6 1 Illustris
"""

def get_smooth_density(D, fwhm, Lbox, N_dim):
    """
    Smooth density map D ((0, Lbox] and N_dim^2 cells) with Gaussian beam of FWHM
    """
    kstep = 2*np.pi/(N_dim*np.pi/180*pixsizedeg)
    #karr = np.fft.fftfreq(N_dim, d=Lbox/(2*np.pi*N_dim)) # physical (not correct)
    karr = np.fft.fftfreq(N_dim, d=Lboxdeg*np.pi/180./(2*np.pi*N_dim)) # angular
    print("kstep = ", kstep, karr[1]-karr[0]) # N_dim/d gives kstep
    
    # fourier transform the map and apply gaussian beam
    D = D.astype(np.float32)
    dfour = scipy.fft.fftn(D, workers=-1)
    dksmo = np.zeros((N_dim, N_dim), dtype=np.complex64)
    ksq = np.zeros((N_dim, N_dim), dtype=np.complex64)
    ksq[:, :] = karr[None, :]**2+karr[:,None]**2
    dksmo[:, :] = gauss_beam(ksq, fwhm)*dfour
    drsmo = np.real(scipy.fft.ifftn(dksmo, workers=-1))

    return drsmo 

def gauss_beam(ellsq, fwhm):
    """
    Gaussian beam of size fwhm
    """
    tht_fwhm = np.deg2rad(fwhm/60.)
    return np.exp(-0.5*(tht_fwhm**2.)*(ellsq)/(8.*np.log(2.)))

# simulation choices
save_dir = "/freya/ptmp/mpa/boryanah/data_sz/" # virgo
#save_dir = "/n/holystore01/LABS/hernquist_lab/Everyone/bhadzhiyska/SZ_TNG/" # cannon
#save_dir = "/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/SZ_TNG/" 
dirs = ['xy']#, 'yz', 'zx']
want_show = False
want_dm = int(sys.argv[2])

# simulation choices
#sim_name = "MTNG"
#sim_name = "TNG300"
#sim_name = "Illustris"
#sim_name = "TNG100"
sim_name = sys.argv[3]
if sim_name == "TNG300":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_tng205.txt"), skiprows=1, unpack=True)
    Lbox = 205.
    save_dir += "TNG300/"
if sim_name == "TNG100":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_tng75.txt"), skiprows=1, unpack=True)
    Lbox = 75.
    save_dir += "TNG100/"
elif sim_name == "Illustris":
    snaps, _, zs = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_orig.txt"), skiprows=1, unpack=True)
    Lbox = 75.
    save_dir += "Illustris/"
elif sim_name == "MNTG":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_mtng.txt"), skiprows=1, unpack=True)
    Lbox = 500.
snaps = snaps.astype(int)
if sim_name == "Illustris":
    snapshots = [105, 98, 92, 88] # Illustris
elif sim_name == "TNG300":
    snapshots = [69, 62, 56, 52]#[84, 72, 63]#[78, 91, 59]#[67]#[63, 59, 56, 53, 50]
# 99, 91, 84, 78, 72, 67, 63, 59, 56, 53, 50
# 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.79, 0.89, 1.

# gaussian beam
#fwhm = 1.3 #arcmin 2009.05557 not ideal
#fwhm = 2.1 #arcmin 2009.05557
fwhm = float(sys.argv[1])

if sim_name ==  "Illustris":
    # define cosmology
    h = 0.704
    cosmo = FlatLambdaCDM(H0=h*100., Om0=0.2726, Tcmb0=2.725)
    N_dim = 2000
else:
    # define cosmology
    h = 0.6774
    cosmo = FlatLambdaCDM(H0=h*100., Om0=0.3089, Tcmb0=2.725)
if sim_name == "Illustris" or sim_name == "TNG100":
    N_dim = 2000
else:
    N_dim = 10000

for snapshot in snapshots:
    redshift = zs[snaps == snapshot]
    a = 1./(1+redshift)
    print("redshift = ", redshift)

    # compute angular distance
    d_L = cosmo.luminosity_distance(redshift).to(u.Mpc).value
    d_A = d_L/(1.+redshift)**2 # dA = dL/(1+z)^2 # Mpc
    d_A *= h
    print("d_A =", d_A) # Mpc/h

    # box size in degrees
    Lboxdeg = np.sqrt((a*Lbox)**2/d_A**2*(180./np.pi)**2)
    pixsizedeg = Lboxdeg/N_dim
    print("Lboxdeg = ", Lboxdeg)

    # for each of three projections
    for i in range(len(dirs)):
        if want_dm:
           # load dm map
            dm_xy = np.load(f"{save_dir}/dm_{dirs[i]:s}_snap_{snapshot:d}.npy")
            print("dm_xy = ", dm_xy[:10])

            # smooth with beam
            dm_beam_xy = get_smooth_density(dm_xy, fwhm, Lbox, N_dim)
            print("obtained smooth version")
            np.save(f"{save_dir}/dm_beam{fwhm:.1f}_{dirs[i]:s}_snap_{snapshot:d}.npy", dm_beam_xy)

        else:
            # load tSZ map
            Y_xy = np.load(f"{save_dir}/Y_compton_{dirs[i]:s}_snap_{snapshot:d}.npy")
            print("Y_xy = ", Y_xy[:10])

            # smooth with beam
            Y_beam_xy = get_smooth_density(Y_xy, fwhm, Lbox, N_dim)
            print("obtained smooth version")
            np.save(f"{save_dir}/Y_compton_beam{fwhm:.1f}_{dirs[i]:s}_snap_{snapshot:d}.npy", Y_beam_xy)
            std_Y_xy = Y_beam_xy.std()
            mean_Y_xy = Y_beam_xy.mean()

            if want_show:
                # plot and save unbeamed
                plt.figure(figsize=(16,14))
                plt.imshow(np.log10(1+Y_xy), vmin=np.log10(1+mean_Y_xy-4.*std_Y_xy), vmax=np.log10(1+mean_Y_xy+4.*std_Y_xy))
                plt.savefig(f"figs/Y_{dirs[i]:s}_snap{snapshot}.png")
                plt.close()

                # plot and save beamed
                plt.figure(figsize=(16,14))
                plt.imshow(np.log10(1+Y_beam_xy), vmin=np.log10(1+mean_Y_xy-4.*std_Y_xy), vmax=np.log10(1+mean_Y_xy+4.*std_Y_xy))
                plt.savefig(f"figs/Y_beam_{dirs[i]:s}_snap{snapshot}.png")
                plt.close()

            # load kSZ map
            b_xy = np.load(f"{save_dir}/b_{dirs[i]:s}_snap_{snapshot:d}.npy")
            print("b_xy = ", b_xy[:10])

            # smooth with beam
            b_beam_xy = get_smooth_density(b_xy, fwhm, Lbox, N_dim)
            print("obtained smooth version")
            np.save(f"{save_dir}/b_beam{fwhm:.1f}_{dirs[i]:s}_snap_{snapshot:d}.npy", b_beam_xy)
            std_b_xy = b_beam_xy.std()
            mean_b_xy = b_beam_xy.mean() 

            if want_show:
                # plot and save unbeamed
                plt.figure(figsize=(16,14))
                plt.imshow((1+b_xy), vmin=(1+mean_b_xy-4.*std_b_xy), vmax=(1+mean_b_xy+4.*std_b_xy))
                plt.savefig(f"figs/b_{dirs[i]:s}_snap{snapshot}.png")
                plt.close()

                # plot and save beamed
                plt.figure(figsize=(16,14))
                plt.imshow((1+b_beam_xy), vmin=(1+mean_b_xy-4.*std_b_xy), vmax=(1+mean_b_xy+4.*std_b_xy))
                plt.savefig(f"figs/b_beam_{dirs[i]:s}_snap{snapshot}.png")
                plt.close()

            # load tau map
            tau_xy = np.load(f"{save_dir}/tau_{dirs[i]:s}_snap_{snapshot:d}.npy")
            print("tau_xy = ", tau_xy[:10])

            # smooth with beam
            tau_beam_xy = get_smooth_density(tau_xy, fwhm, Lbox, N_dim)
            print("obtained smooth version")
            np.save(f"{save_dir}/tau_beam{fwhm:.1f}_{dirs[i]:s}_snap_{snapshot:d}.npy", tau_beam_xy)
            std_tau_xy = tau_beam_xy.std()
            mean_tau_xy = tau_beam_xy.mean() 

            if want_show:
                # plot and save unbeamed
                plt.figure(figsize=(16,14))
                plt.imshow((1+tau_xy), vmin=(1+mean_tau_xy-4.*std_tau_xy), vmax=(1+mean_tau_xy+4.*std_tau_xy))
                plt.savefig(f"figs/tau_{dirs[i]:s}_snap{snapshot}.png")
                plt.close()

                # plot and save beamed
                plt.figure(figsize=(16,14))
                plt.imshow((1+tau_beam_xy), vmin=(1+mean_tau_xy-4.*std_tau_xy), vmax=(1+mean_tau_xy+4.*std_tau_xy))
                plt.savefig(f"figs/tau_beam_{dirs[i]:s}_snap{snapshot}.png")
                plt.close()

        gc.collect()
