"""
Script for saving fields of interest for each gas cell to be used in creating SZ maps
"""
import numpy as np
import h5py
from tools import numba_tsc_3D
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

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
kpc_to_cm = ((1.*u.kpc).to(u.cm)).value # cm
solar_mass = 1.989e33 # g

# sim params (same for MTNG and TNG)
h = 67.74/100.
unit_mass = 1.e10*(solar_mass/h)
unit_dens = 1.e10*(solar_mass/h)/(kpc_to_cm/h)**3 # g/cm**3 # note density has units of h in it
unit_vol = (kpc_to_cm/h)**3

# simulation info
#basePath = "/virgotng/mpa/MTNG/Hydro-Arepo/MTNG-L500-4320-A/output/"; snapshot = 179; n_chunks = 640; z = 1. # MTNG
basePath = "/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/L205n2500TNG/output"; snapshot = 99; n_chunks = 600; z = 0. # TNG
#save_dir = "/freya/ptmp/mpa/boryanah/data_sz/" # virgo
save_dir = "/n/holystore01/LABS/hernquist_lab/Everyone/bhadzhiyska/SZ_TNG/" # cannon
PartType = 'PartType0' # gas cells
a = 1./(1+z)

# fields to load
fields = ['InternalEnergy', 'ElectronAbundance', 'Density', 'Masses', 'Coordinates', 'Velocities']

# impose minimum temperature threshold to save space
Tmin = 0. # 1.e4, 1.e6

# loop over each chunk in the simulation
for i in range(0, n_chunks):
    print("chunk = ", i)

    # read positions of DM particles
    hfile = h5py.File(basePath+f'snapdir_{snapshot:03d}/snapshot_{snapshot:03d}.{i:d}.hdf5')[PartType]
    #print(list(hfile.keys()))

    # select all fields of interest
    C = hfile['Coordinates'][:]
    EA = hfile['ElectronAbundance'][:]
    IE = hfile['InternalEnergy'][:]
    D = hfile['Density'][:]
    M = hfile['Masses'][:]
    C = hfile['Coordinates'][:]
    V = hfile['Velocities'][:]
    
    # for each cell, compute its total volume (gas mass by gas density) and convert density units
    dV = M/D # cMpc/h^3 (MTNG) or ckpc/h^3 (TNG)
    D *= unit_dens # g/cm^3 # True for TNG and mixed for MTNG because of unit difference

    # obtain electron temperature, electron number density and velocity
    Te = (gamma - 1.)*IE/k_B * 4*m_p/(1 + 3*X_H + 4*X_H*EA) * unit_c # K
    ne = EA*X_H*D/m_p # cm^-3 # True for TNG and mixed for MTNG because of unit difference
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
