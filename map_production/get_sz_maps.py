"""
Script for obtaining SZ maps in three projections
"""
import time
import os

import numpy as np
import h5py

from tools import numba_tsc_3D, hist2d_numba_seq
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u


# physics constants in cgs
gamma = 5/3. # unitless
k_B = 1.3807e-16 # cgs (erg/K)
m_p = 1.6726e-24 # g
unit_c = 1.023**2*1.e10 # see faq km/s to pc/Myr
X_H = 0.76 # unitless
sigma_T = 6.6524587158e-29*1.e2**2 # cm^2
m_e = 9.10938356e-28 # g
c = 29979245800. # cm/s
const = k_B*sigma_T/(m_e*c**2) # cgs (cm^2/K)
kpc_to_cm = ((1.*u.kpc).to(u.cm)).value # cm
solar_mass = 1.989e33 # g

# sim params
h = 67.74/100.
unit_mass = 1.e10*(solar_mass/h)
unit_dens = 1.e10*(solar_mass/h)/(kpc_to_cm/h)**3 # g/cm**3 # note density has units of h in it
unit_vol = (kpc_to_cm/h)**3 # cancels unit dens division from previous, so h doesn't matter neither does kpc

# simulation choices
#sim_name = "MTNG"
sim_name = "TNG300"
if sim_name == "TNG300":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_tng205.txt"), skiprows=1, unpack=True)
elif sim_name == "MNTG":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_mtng.txt"), skiprows=1, unpack=True)
snaps = snaps.astype(int)
snapshots = [84, 72, 63]#, 56, 53, 50]#[78, 91, 59]#[67]#[91, 84, 78, 72, 67, 63, 59, 56, 53, 50]
# 99, 91, 84, 78, 72, 67, 63, 59, 56, 53, 50
# 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.79, 0.89, 1.

# simulation info
if sim_name == "TNG300":
    n_chunks = 600
    #save_dir = "/n/holystore01/LABS/hernquist_lab/Everyone/bhadzhiyska/SZ_TNG/" # cannon
    save_dir = "/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/SZ_TNG/" # cannon
    Lbox = 205000. # kpc/h
    Lbox_hkpc = Lbox # ckpc/h
elif sim_name == "MNTG":
    n_chunks = 640
    save_dir = "/freya/ptmp/mpa/boryanah/data_sz/" # virgo
    Lbox = 500. # Mpc/h
    Lbox_hkpc = Lbox*1000. # ckpc/h
PartType = 'PartType0' # gas cells
cosmo = FlatLambdaCDM(H0=h*100., Om0=0.3089, Tcmb0=2.725)

# fields to load
fields = ['InternalEnergy', 'ElectronAbundance', 'Density', 'Masses', 'Coordinates', 'Velocities']

# for the histograming
nbins = 10001
ranges = ((0., Lbox),(0., Lbox))
nbins2d = (nbins-1, nbins-1)
nbins2d = np.asarray(nbins2d).astype(np.int64)
ranges = np.asarray(ranges).astype(np.float64)

# bins
bins = np.linspace(0, Lbox, nbins)
binc = (bins[1:]+bins[:-1])*0.5

for snapshot in snapshots:
    z = zs[snaps == snapshot]
    a = 1./(1+z)
    print("redshift = ", z)

    # angular distance
    d_L = cosmo.luminosity_distance(z).to(u.kpc).value
    d_A = d_L/(1.+z)**2 # dA = dL/(1+z)^2 # kpc
    print("angular size = Mpc/h", d_A*h/1000.) # Mpc/h 1150
    d_A *= kpc_to_cm # cm should be the right unit

    # initialize empty maps
    y_xy = np.zeros((nbins-1, nbins-1))
    y_yz = np.zeros((nbins-1, nbins-1))
    y_zx = np.zeros((nbins-1, nbins-1))
    b_xy = np.zeros((nbins-1, nbins-1))
    b_yz = np.zeros((nbins-1, nbins-1))
    b_zx = np.zeros((nbins-1, nbins-1))
    tau_xy = np.zeros((nbins-1, nbins-1))
    tau_yz = np.zeros((nbins-1, nbins-1))
    tau_zx = np.zeros((nbins-1, nbins-1))

    # loop over each chunk
    for i in range(0, n_chunks):
        print("chunk = ", i, end='\r')

        # read saved fields of interest
        Te = np.load(f"{save_dir}/temperature_chunk_{i:d}_snap_{snapshot:d}.npy") # K
        ne = np.load(f"{save_dir}/number_density_chunk_{i:d}_snap_{snapshot:d}.npy") # cm^-3 # True for TNG and mixed for MTNG because of unit difference
        Ve = np.load(f"{save_dir}/velocity_chunk_{i:d}_snap_{snapshot:d}.npy")*1.e5 # cm/s from km/s
        dV = np.load(f"{save_dir}/volume_chunk_{i:d}_snap_{snapshot:d}.npy") # cMpc/h^3 (MTNG) and ckpc/h^3 (TNG)
        C = np.load(f"{save_dir}/position_chunk_{i:d}_snap_{snapshot:d}.npy") # cMpc/h (MTNG) and ckpc/h (TNG)
    
        # compute the contribution to the y and b signals of each cell
        # ne*dV cancel unit length of simulation and unit_vol converts ckpc/h^3 to cm^3
        # both should be unitless (const*Te/d_A**2 is cm^2/cm^2; sigma_T/d_A^2 is unitless)
        dY = const*(ne*Te*dV)*unit_vol/(a*Lbox_hkpc*(kpc_to_cm/h))**2.#d_A**2 
        b = sigma_T*(ne[:, None]*(Ve/c)*dV[:, None])*unit_vol/(a*Lbox_hkpc*(kpc_to_cm/h))**2.#d_A**2
        tau = sigma_T*(ne*dV)*unit_vol/(a*Lbox_hkpc*(kpc_to_cm/h))**2.#d_A**2

        # flatten out in each of three directions (10 times faster than histogramdd
        Dxy = hist2d_numba_seq(np.array([C[:, 0], C[:, 1]]), bins=nbins2d, ranges=ranges, weights=dY)
        Dyz = hist2d_numba_seq(np.array([C[:, 1], C[:, 2]]), bins=nbins2d, ranges=ranges, weights=dY)
        Dzx = hist2d_numba_seq(np.array([C[:, 2], C[:, 0]]), bins=nbins2d, ranges=ranges, weights=dY)

        # coadd the contribution from this chunk to the three projections
        y_xy += Dxy
        y_yz += Dyz
        y_zx += Dzx

        # flatten out in each of three directions
        Dxy = hist2d_numba_seq(np.array([C[:, 0], C[:, 1]]), bins=nbins2d, ranges=ranges, weights=b[:, 2])
        Dyz = hist2d_numba_seq(np.array([C[:, 1], C[:, 2]]), bins=nbins2d, ranges=ranges, weights=b[:, 0])
        Dzx = hist2d_numba_seq(np.array([C[:, 2], C[:, 0]]), bins=nbins2d, ranges=ranges, weights=b[:, 1])

        # coadd the contribution from this chunk to the three projections
        b_xy += Dxy
        b_yz += Dyz
        b_zx += Dzx

        # flatten out in each of three directions
        Dxy = hist2d_numba_seq(np.array([C[:, 0], C[:, 1]]), bins=nbins2d, ranges=ranges, weights=tau)
        Dyz = hist2d_numba_seq(np.array([C[:, 1], C[:, 2]]), bins=nbins2d, ranges=ranges, weights=tau)
        Dzx = hist2d_numba_seq(np.array([C[:, 2], C[:, 0]]), bins=nbins2d, ranges=ranges, weights=tau)

        # coadd the contribution from this chunk to the three projections
        tau_xy += Dxy
        tau_yz += Dyz
        tau_zx += Dzx

    # save tSZ maps
    np.save(f"{save_dir}/Y_compton_xy_snap_{snapshot:d}.npy", y_xy)
    np.save(f"{save_dir}/Y_compton_yz_snap_{snapshot:d}.npy", y_yz)
    np.save(f"{save_dir}/Y_compton_zx_snap_{snapshot:d}.npy", y_zx)
    
    # save kSZ maps
    np.save(f"{save_dir}/b_xy_snap_{snapshot:d}.npy", b_xy)
    np.save(f"{save_dir}/b_yz_snap_{snapshot:d}.npy", b_yz)
    np.save(f"{save_dir}/b_zx_snap_{snapshot:d}.npy", b_zx)

    # save tau maps
    np.save(f"{save_dir}/tau_xy_snap_{snapshot:d}.npy", tau_xy)
    np.save(f"{save_dir}/tau_yz_snap_{snapshot:d}.npy", tau_yz)
    np.save(f"{save_dir}/tau_zx_snap_{snapshot:d}.npy", tau_zx)
