# fix cosmology, h factors and a factors; light cone coordinates? missing tau
import time

import numpy as np
import h5py

from tools import numba_tsc_3D, hist2d_numba_seq
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u


# physics constants in cgs
gamma = 5/3.
k_B = 1.3807e-16 # cgs
m_p = 1.6726e-24 # g
unit_c = 1.023**2*1.e10
X_H = 0.76
sigma_T = 6.6524587158e-29*1.e2**2 # cm^2
m_e = 9.10938356e-28 # g
c = 29979245800. # cm/s
const = k_B*sigma_T/(m_e*c**2) # cgs
kpc_to_cm = ((1.*u.kpc).to(u.cm)).value # cm
solar_mass = 1.989e33 # g

# sim params  I think that's it
h = 67.74/100.
unit_mass = 1.e10*(solar_mass/h)
unit_dens = 1.e10*(solar_mass/h)/(kpc_to_cm/h)**3 # g/cm**3 # note density has units of h in it
unit_vol = (kpc_to_cm/h)**3

# path to simulation
basePath = "/virgo/simulations/MTNG/Hydro-Arepo/MTNG-L500-4320-A/"; type_sim = "_fp"; snapshot = 179; n_chunks = 640; z = 1.
#basePath = "/virgo/simulations/MTNG/DM-Gadget4/MTNG-L500-4320-A/"; type_sim = "_dm"; snapshot = 184; n_chunks = 128; z = 1.

# sim info
n_total = 4320**3
Lbox = 500. # Mpc/h
PartType = 'PartType0'
a = 1./(1+z)

# angular distance
redshift = z
cosmo = FlatLambdaCDM(H0=h*100., Om0=0.3089, Tcmb0=2.725)
d_L = cosmo.luminosity_distance(redshift).to(u.kpc).value
d_A = d_L/(1.+redshift)**2 # dA = dL/(1+z)^2 # kpc
d_A *= kpc_to_cm # cm eq. to 1150 Mpc/h
#d_A *= kpc_to_cm/1.e3 # cm # attention used in the initial maps with z = 1 (forgot the division by speed of light)

# for the histograming
nbins = 10001
ranges = ((0., Lbox),(0., Lbox))
nbins2d = (nbins-1, nbins-1)
nbins2d = np.asarray(nbins2d).astype(np.int64)
ranges = np.asarray(ranges).astype(np.float64)

bins = np.linspace(0, Lbox, nbins)
binc = (bins[1:]+bins[:-1])*0.5
y_xy = np.zeros((nbins-1, nbins-1))
y_yz = np.zeros((nbins-1, nbins-1))
y_zx = np.zeros((nbins-1, nbins-1))
for i in range(0, n_chunks):
    print("chunk = ", i, end='\r')

    #Te = np.load(f"/freya/ptmp/mpa/boryanah/data_sz/temperature_chunk_{i:d}_snap_{snapshot:d}.npy")
    ne = np.load(f"/freya/ptmp/mpa/boryanah/data_sz/number_density_chunk_{i:d}_snap_{snapshot:d}.npy") # cm^-3 #not entirely cause of kpc -> Mpc switch but it oche
    Ve = np.load(f"/freya/ptmp/mpa/boryanah/data_sz/velocity_chunk_{i:d}_snap_{snapshot:d}.npy")*1.e5 # cm/s from km/s
    dV = np.load(f"/freya/ptmp/mpa/boryanah/data_sz/volume_chunk_{i:d}_snap_{snapshot:d}.npy") # Mpc/h^3 unitvol cancels ne units
    C = np.load(f"/freya/ptmp/mpa/boryanah/data_sz/position_chunk_{i:d}_snap_{snapshot:d}.npy")
    # read positions of DM particles

    #dY = const*(ne*Te*dV)*unit_vol/d_A**2 # unit vol changes dV to cm^3
    b = sigma_T*(ne[:, None]*(Ve/c)*dV[:, None])*unit_vol/d_A**2 # unit vol changes dV to cm^3
    #print(b.max())
    
    Dxy = hist2d_numba_seq(np.array([C[:, 0], C[:, 1]]), bins=nbins2d, ranges=ranges, weights=b[:, 2])
    Dyz = hist2d_numba_seq(np.array([C[:, 1], C[:, 2]]), bins=nbins2d, ranges=ranges, weights=b[:, 0])
    Dzx = hist2d_numba_seq(np.array([C[:, 2], C[:, 0]]), bins=nbins2d, ranges=ranges, weights=b[:, 1])
    
    y_xy += Dxy
    y_yz += Dyz
    y_zx += Dzx

np.save("data_maps_sz/b_xy", y_xy)
np.save("data_maps_sz/b_yz", y_yz)
np.save("data_maps_sz/b_zx", y_zx)
