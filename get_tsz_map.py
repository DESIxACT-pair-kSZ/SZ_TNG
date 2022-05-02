# fix cosmology, h factors and a factors; light cone coordinates?
import time

import numpy as np
import h5py

from tools import numba_tsc_3D, hist2d_numba_seq
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u


# physics constants in cgs
gamma = 5/3.
k_B = 1.3807e-16 # cgs (erg/K)
m_p = 1.6726e-24 # g
unit_c = 1.023**2*1.e10 # see faq km/s to pc/Myr
X_H = 0.76
sigma_T = 6.6524587158e-29*1.e2**2 # cm^2
m_e = 9.10938356e-28 # g
c = 29979245800. # cm/s
const = k_B*sigma_T/(m_e*c**2) # cgs (cm^2/K)
kpc_to_cm = ((1.*u.kpc).to(u.cm)).value # cm
solar_mass = 1.989e33 # g

# sim params I think that's it
h = 67.74/100.
unit_mass = 1.e10*(solar_mass/h)
unit_dens = 1.e10*(solar_mass/h)/(kpc_to_cm/h)**3 # g/cm**3 # note density has units of h in it
unit_vol = (kpc_to_cm/h)**3 # cancels unit dens division from previous, so h doesn't matter neither does kpc

# path to simulation
basePath = "/virgo/simulations/MTNG/Hydro-Arepo/MTNG-L500-4320-A/"; type_sim = "_fp"; snapshot = 179; n_chunks = 640; z = 1.
#basePath = "/virgo/simulations/MTNG/DM-Gadget4/MTNG-L500-4320-A/"; type_sim = "_dm"; snapshot = 184; n_chunks = 128; z = 1.

# sim info
n_total = 4320**3
Lbox = 500. # Mpc/h
PartType = 'PartType0'
a = 1./(1+z)


# angular distance
redshift = z # if I choose 0.01 I get the bulk right answer
cosmo = FlatLambdaCDM(H0=h*100., Om0=0.3089, Tcmb0=2.725)
d_L = cosmo.luminosity_distance(redshift).to(u.kpc).value
d_A = d_L/(1.+redshift)**2 # dA = dL/(1+z)^2 # kpc
print(d_A*h/1000.) # Mpc/h 1150
d_A *= kpc_to_cm # cm should be the right one, I think
# attention!!! I think for the files I produced I used z = 1 and d_A/1000.

# fields to load
fields = ['InternalEnergy', 'ElectronAbundance', 'Density', 'Masses', 'Coordinates', 'Velocities']

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

    Te = np.load(f"/freya/ptmp/mpa/boryanah/data_sz/temperature_chunk_{i:d}_snap_{snapshot:d}.npy")
    ne = np.load(f"/freya/ptmp/mpa/boryanah/data_sz/number_density_chunk_{i:d}_snap_{snapshot:d}.npy")
    #Ve = np.load(f"/freya/ptmp/mpa/boryanah/data_sz/velocity_chunk_{i:d}_snap_{snapshot:d}.npy")
    dV = np.load(f"/freya/ptmp/mpa/boryanah/data_sz/volume_chunk_{i:d}_snap_{snapshot:d}.npy") # cMpc/h^3
    C = np.load(f"/freya/ptmp/mpa/boryanah/data_sz/position_chunk_{i:d}_snap_{snapshot:d}.npy")
    # read positions of DM particles
    
    #dY = 1./d_A**2*const*(ne*Te*dV)*unit_vol # unit vol changes dV to cm^3
    dY = const*(ne*Te*dV)*unit_vol/d_A**2 # unit vol changes dV to cm^3; should be unitless
    #print("ne, Te, dV = ", ne[:10], Te[:10], dV[:10])
    
    #dY = dY.astype(np.float32)
    #C = C.astype(np.float32)

    # 10 times faster than the other ones
    Dxy = hist2d_numba_seq(np.array([C[:, 0], C[:, 1]]), bins=nbins2d, ranges=ranges, weights=dY)
    Dyz = hist2d_numba_seq(np.array([C[:, 1], C[:, 2]]), bins=nbins2d, ranges=ranges, weights=dY)
    Dzx = hist2d_numba_seq(np.array([C[:, 2], C[:, 0]]), bins=nbins2d, ranges=ranges, weights=dY)
    
    """
    Dxy, edges = np.histogramdd(C[:, :2], bins=nbins-1, range=ranges, weights=dY)
    Dyz, edges = np.histogramdd(C[:, 1:], bins=nbins-1, range=ranges, weights=dY)
    Dzx, edges = np.histogramdd(np.transpose([C[:, 2], C[:, 0]]), bins=nbins-1, range=[[0, Lbox],[0, Lbox]], weights=dY)

    Dxy, xedges, yedges = np.histogram2d(C[:, 0], C[:, 1], bins=nbins-1, range=[[0, Lbox],[0, Lbox]], weights=dY)
    Dyz, xedges, yedges = np.histogram2d(C[:, 1], C[:, 2], bins=nbins-1, range=[[0, Lbox],[0, Lbox]], weights=dY)
    Dzx, xedges, yedges = np.histogram2d(C[:, 2], C[:, 0], bins=nbins-1, range=[[0, Lbox],[0, Lbox]], weights=dY)
    """
    
    y_xy += Dxy
    y_yz += Dyz
    y_zx += Dzx

np.save("data_maps_sz/Y_compton_xy", y_xy)
np.save("data_maps_sz/Y_compton_yz", y_yz)
np.save("data_maps_sz/Y_compton_zx", y_zx)
