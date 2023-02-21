"""
Script for saving fields of interest for each gas cell to be used in creating SZ maps

Cannon has:
snapdir_001/ snapdir_003/ snapdir_006/ snapdir_013/ snapdir_021/ snapdir_033/ snapdir_050/ snapdir_078/
snapdir_002/ snapdir_004/ snapdir_008/ snapdir_017/ snapdir_025/ snapdir_040/ snapdir_067/ snapdir_099/

"""
import gc
import os
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

# simulation choices
#sim_name = "MTNG"
#sim_name = "TNG300"
sim_name = "CAMELS";which_sim = "EX_3"
if sim_name == "TNG300":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_tng205.txt"), skiprows=1, unpack=True)
elif sim_name == "CAMELS":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_camels.txt"), skiprows=1, unpack=True)
elif sim_name == "MNTG":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_mtng.txt"), skiprows=1, unpack=True)
snaps = snaps.astype(int)
# 99, 91, 84, 78, 72, 67, 63, 59, 56, 53, 50
# 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.79, 0.89, 1.

snapshots = [78, 72, 67, 63, 59, 56, 53, 50]
redshifts = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
snapshots = np.zeros(len(redshifts), dtype=int)
for i, red in enumerate(redshifts):
    ind = np.argmin(np.abs(red - zs))
    snapshots[i] = snaps[ind]
print(snapshots)

# simulation info
if sim_name == "TNG300":
    basePath = "/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/L205n2500TNG/output/"
    n_chunks = 600
    #save_dir = "/n/holystore01/LABS/hernquist_lab/Everyone/bhadzhiyska/SZ_TNG/" # cannon
    save_dir = "/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/SZ_TNG/" # cannon
elif sim_name == "CAMELS":
    basePath = f"/n/holylfs05/LABS/hernquist_lab/Users/bhadzhiyska/CAMELS/{which_sim}/"
    n_chunks = 1
    #save_dir = "/n/holystore01/LABS/hernquist_lab/Everyone/bhadzhiyska/SZ_TNG/" # cannon
    save_dir = f"/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/SZ_TNG/CAMELS/{which_sim}/" # cannon
elif sim_name == "MNTG":
    basePath = "/virgotng/mpa/MTNG/Hydro-Arepo/MTNG-L500-4320-A/output/"
    n_chunks = 640
    save_dir = "/freya/ptmp/mpa/boryanah/data_sz/" # virgo
PartType = 'PartType0' # gas cells
os.makedirs(save_dir, exist_ok=True)

# fields to load
fields = ['InternalEnergy', 'ElectronAbundance', 'Density', 'Masses', 'Coordinates', 'Velocities']

# impose minimum temperature threshold to save space
Tmin = 0. # 1.e4, 1.e6

# loop over each chunk in the simulation
for snapshot in snapshots:
    z = zs[snaps == snapshot]
    a = 1./(1+z)
    print("redshift = ", z)

    for i in range(0, n_chunks): 
        print("chunk = ", i)

        # read positions of DM particles
        if sim_name == "TNG300":
            hfile = h5py.File(basePath+f'snapdir_{snapshot:03d}/snap_{snapshot:03d}.{i:d}.hdf5')[PartType]
        elif sim_name == "CAMELS":
            hfile = h5py.File(basePath+f'snap_{snapshot:03d}.hdf5')[PartType]
        elif sim_name == "MNTG":
            hfile = h5py.File(basePath+f'snapdir_{snapshot:03d}/snapshot_{snapshot:03d}.{i:d}.hdf5')[PartType]
        #print(list(hfile.keys()))

        # select all fields of interest
        C = hfile['Coordinates'][:]
        EA = hfile['ElectronAbundance'][:]
        IE = hfile['InternalEnergy'][:]
        D = hfile['Density'][:]
        M = hfile['Masses'][:]
        V = hfile['Velocities'][:]
    
        # for each cell, compute its total volume (gas mass by gas density) and convert density units
        dV = M/D # cMpc/h^3 (MTNG) or ckpc/h^3 (TNG)
        D *= unit_dens # g/ccm^3 # True for TNG and mixed for MTNG because of unit difference

        # obtain electron temperature, electron number density and velocity
        Te = (gamma - 1.)*IE/k_B * 4*m_p/(1 + 3*X_H + 4*X_H*EA) * unit_c # K
        ne = EA*X_H*D/m_p # ccm^-3 # True for TNG and mixed for MTNG because of unit difference
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
        np.save(f"{save_dir}/temperature_chunk_{i:d}_snap_{snapshot:d}.npy", Te.astype(np.float32))
        np.save(f"{save_dir}/number_density_chunk_{i:d}_snap_{snapshot:d}.npy", ne.astype(np.float32))
        np.save(f"{save_dir}/velocity_chunk_{i:d}_snap_{snapshot:d}.npy", Ve.astype(np.float32))
        np.save(f"{save_dir}/volume_chunk_{i:d}_snap_{snapshot:d}.npy", dV.astype(np.float32))
        np.save(f"{save_dir}/position_chunk_{i:d}_snap_{snapshot:d}.npy", C.astype(np.float32))
        np.save(f"{save_dir}/density_chunk_{i:d}_snap_{snapshot:d}.npy", D.astype(np.float32))
        del M, D, EA, IE, Te, ne, Ve, dV, C, choice; gc.collect()
