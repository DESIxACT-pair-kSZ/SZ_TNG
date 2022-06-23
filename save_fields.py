"""
Script for saving fields of interest for each gas cell to be used in creating SZ maps
"""
import numpy as np
import h5py
from tools import numba_tsc_3D
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import argparse
import os 

# command line arguments -- latin hypercube number, initial and final snapshot number
# note: final snapshot number is the highest redshift for lightcone
#ex. save_fields.py -lh 0 -si 32 -sf 31 

parser = argparse.ArgumentParser()
parser.add_argument('--LHnum','-lh', type=str)
parser.add_argument('--snapinit','-si', type=int)
parser.add_argument('--snapfinal','-sf', type=int)
args = parser.parse_args()
LHnum = args.LHnum
snapinit = args.snapinit
snapfinal = args.snapfinal

# check to see that LH and snap nums are valid
LHlist = [str(i) for i in range(1000)]
snaplist = np.arange(34)

if LHnum not in LHlist:
    print('LH number out of bounds')
save_dir = "/global/cscratch1/sd/kjc268/CAMELS_TSZ/LH_{}".format(LHnum) 
# make subdirectory for the specified LH if it doesn't exist
if not os.path.isdir(save_dir): 
    os.mkdir(save_dir)

if snapinit not in snaplist: 
    print('snapshot number out of bounds')
if snapfinal in snaplist: 
    if snapfinal > snapinit: 
        print('final snapshot number must be less than initial snapshot number')
    elif snapfinal == snapinit:
        print('no lightcone, single redshift maps')
else: 
    snapfinal = snapinit 
    print('no lightcone, single redshift maps')

# physics constants in cgs
gamma = 5/3. # unitless
k_B = 1.3807e-16 # cgs (erg/K)
m_p = 1.6726e-24 # g
unit_c = 1.e10 # TNG faq is wrong (see README.md)
X_H = 0.76 # unitless
sigma_T = 6.6524587158e-29*1.e2**2 # cm^2
m_e = 9.10938356e-28 # g
c = 29979245800. # cm/s
const = k_B*sigma_T/(m_e*c**2) # cgs (cm^2/K)
kpc_to_cm = ((1.*u.kpc).to(u.cm)).value # cm / kpc

snaps = np.arange(snapfinal,snapinit+1)
PartType = 'PartType0' # gas cell
# fields to load
fields = ['InternalEnergy', 'ElectronAbundance', 'Density', 'Masses', 'Coordinates', 'Velocities']
# impose minimum temperature threshold to save space
Tmin = 0. # 1.e4, 1.e6

# loop over each snapshot
for snapshot in snaps:
    print("snapshot =", snapshot)
    basePath = "/global/cscratch1/sd/kjc268/CAMELS_ITNG/LH_{}/snap_{}.hdf5".format(LHnum,str(snapshot).zfill(3))
    n_chunks = h5py.File(basePath)['Header'].attrs['NumFilesPerSnapshot']
    z = h5py.File(basePath)['Header'].attrs['Redshift'] # TNG
    a = 1./(1+z)
 
    # sim params (same for MTNG and TNG)
    h = h5py.File(basePath)['Header'].attrs['HubbleParam']
    solar_mass = 1.989e33 # g/msun
    unit_mass = solar_mass*1e10/h # h g/ 10^10 msun
    unit_vol = (kpc_to_cm/h)**3 # (cm/kpc/h)^3
    unit_dens = unit_mass/unit_vol # h g/(10^10 msun) / (cm/kpc/h)^3 

    # loop over each chunk in the simulation
    for i in range(0, n_chunks):
        print("chunk = ", i)

        # read positions of DM particles
        hfile = h5py.File(basePath)[PartType]

        # select all fields of interest
        C = hfile['Coordinates'][:] # ckpc/h
        EA = hfile['ElectronAbundance'][:] # dimensionless
        IE = hfile['InternalEnergy'][:] # (km/s)^2
        D = hfile['Density'][:] # (10^10 msun / h) / (ckpc/h)^3 
        M = hfile['Masses'][:] # (10^10 msun / h)
        V = hfile['Velocities'][:] # sqrt(a) km/s

        # for each cell, compute its total volume (gas mass by gas density) and convert density units
        dV = M/D # (ckpc/h)^3 (TNG)
        D *= unit_dens # g/ccm^3 (comoving cm)

        # obtain electron temperature, electron number density and velocity
        Te = (gamma - 1.)*IE/k_B * 4*m_p/(1 + 3*X_H + 4*X_H*EA) * unit_c # K
        ne = EA*X_H*D/m_p # ccm^-3 
        Ve = V*np.sqrt(a) # km/s

        # select cells above certain temperature
        choice = Te > Tmin
        print("percentage above Tmin = ", np.sum(choice)*100./len(choice))

        # make cuts on the fields of interest
        Te = Te[choice]
        ne = ne[choice]
        Ve = Ve[choice]
        dV = dV[choice]
        C = C[choice]
        print("mean temperature = ", np.mean(Te))

        # save all fields of interest
        np.save(f"{save_dir}/temperature_chunk_{i:d}_snap_{snapshot:d}.npy", Te)
        np.save(f"{save_dir}/number_density_chunk_{i:d}_snap_{snapshot:d}.npy", ne)
        np.save(f"{save_dir}/velocity_chunk_{i:d}_snap_{snapshot:d}.npy", Ve)
        np.save(f"{save_dir}/volume_chunk_{i:d}_snap_{snapshot:d}.npy", dV)
        np.save(f"{save_dir}/position_chunk_{i:d}_snap_{snapshot:d}.npy", C)
