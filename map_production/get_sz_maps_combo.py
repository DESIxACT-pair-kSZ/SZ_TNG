"""
full dumps:
51, 69, 80, 94, 129, 151, 179, 214, 237, 264
"""
import sys
import os
import gc

import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy import spatial
import h5py

from tools import numba_tsc_3D, hist2d_numba_seq

import illustris_python as il

from mpi4py import MPI

"""
# MillenniumTNG
mpirun -np 16 python get_sz_maps_combo.py 16 1.0; mpirun -np 16 python get_sz_maps_combo.py 16 0.5; mpirun -np 16 python get_sz_maps_combo.py 16 0.0
#mpirun -np 16 python get_sz_maps_combo.py 16 0.25

python get_sz_maps_combo.py 1 0.5


# SIMBA100
python get_sz_maps_combo.py 1 0.5

# TNG300 (600 chunks)
mpirun -np 30 python get_sz_maps_combo.py 30 0.47; mpirun -np 30 python get_sz_maps_combo.py 30 0.628; mpirun -np 30 python get_sz_maps_combo.py 30 0.791; mpirun -np 30 python get_sz_maps_combo.py 30 0.924

mpirun -np 30 python get_sz_maps_combo.py 30 0.3

# TNG100
mpirun -np 32 python get_sz_maps_combo.py 32 0.47; 

# Illustris (512)
mpirun -np 32 python get_sz_maps_combo.py 32 0.47; mpirun -np 32 python get_sz_maps_combo.py 32 0.628; mpirun -np 32 python get_sz_maps_combo.py 32 0.791; mpirun -np 32 python get_sz_maps_combo.py 32 0.924

# MTNG
mpirun -np 32 python get_sz_maps_combo.py 32 0.; mpirun -np 32 python get_sz_maps_combo.py 32 0.5; mpirun -np 32 python get_sz_maps_combo.py 32 1.0


mpirun -np 32 python get_sz_maps_combo.py 32 0.3 Illustris
mpirun -np 30 python get_sz_maps_combo.py 30 0.3 TNG300

python get_sz_maps_combo.py 1 0.3 SIMBA
"""

myrank = MPI.COMM_WORLD.Get_rank()
n_ranks = int(sys.argv[1])
print(myrank)

# constants cgs
gamma = 5/3.
k_B = 1.3807e-16 # erg/T
m_p = 1.6726e-24 # g
unit_c = 1.023**2*1.e10
X_H = 0.76 
sigma_T = 6.6524587158e-29*1.e2**2 # cm^2
m_e = 9.10938356e-28 # g
c = 29979245800. # cm/s
const = k_B*sigma_T/(m_e*c**2) # cm^2/K
kpc_to_cm = ((1.*u.kpc).to(u.cm)).value # cm
solar_mass = 1.989e33 # g

# map types
map_types = ["tau", "Y_compton", "b"]
#map_types = ["tau"]

# simulation info
#sim_name = "MTNG"
#sim_name = "SIMBA"
#sim_name = "TNG300"
#sim_name = "TNG100"
#sim_name = "Illustris"
sim_name = sys.argv[3]
if sim_name == "TNG300":
    #basePath = "/n/holylfs05/LABS/hernquist_lab/IllustrisTNG/Runs/L205n2500TNG/output/"
    basePath = "/virgotng/universe/IllustrisTNG/L205n2500TNG/output/"
    n_chunks = 600
    #save_dir = "/n/holystore01/LABS/hernquist_lab/Everyone/bhadzhiyska/SZ_TNG/" # cannon
    #save_dir = "/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/SZ_TNG/" # cannon
    save_dir = "/freya/ptmp/mpa/boryanah/data_sz/TNG300/" # virgo
if sim_name == "TNG100":
    basePath = "/virgotng/universe/IllustrisTNG/L75n1820TNG/output/"
    n_chunks = 448
    save_dir = "/freya/ptmp/mpa/boryanah/data_sz/TNG100/" # virgo
elif sim_name == "Illustris":
    basePath = "/virgotng/universe/Illustris/L75n1820FP/output/"
    n_chunks = 512
    save_dir = "/freya/ptmp/mpa/boryanah/data_sz/Illustris/" # virgo
elif sim_name == "CAMELS":
    basePath = f"/n/holylfs05/LABS/hernquist_lab/Users/bhadzhiyska/CAMELS/{which_sim}/"
    n_chunks = 1
    #save_dir = "/n/holystore01/LABS/hernquist_lab/Everyone/bhadzhiyska/SZ_TNG/" # cannon
    save_dir = f"/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/SZ_TNG/CAMELS/{which_sim}/" # cannon
elif sim_name == "SIMBA":
    basePath = "/ptmp/mpa/boryanah/SIMBA100/"
    n_chunks = 1
    save_dir = "/ptmp/mpa/boryanah/SIMBA100/" # virgo
elif sim_name == "MTNG":
    basePath = "/virgotng/mpa/MTNG/Hydro-Arepo/MTNG-L500-4320-A/output/"
    n_chunks = 640
    save_dir = "/freya/ptmp/mpa/boryanah/data_sz/" # virgo
PartType = 'PartType0' # gas cells
os.makedirs(save_dir, exist_ok=True)

# cosmo params
if sim_name == "Illustris":
    h = 0.704
    Omega_m = 0.2726
else:
    h = 67.74/100.
    Omega_m = 0.3089
unit_mass = 1.e10*(solar_mass/h) # g
unit_dens = 1.e10*(solar_mass/h)/(kpc_to_cm/h)**3 # g/cm**3 # note density has units of h in it
unit_vol = (kpc_to_cm/h)**3 # cm^3 cancels unit dens division from previous, so h doesn't matter neither does kpc

# simulation choices
if sim_name == "TNG300":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_tng205.txt"), skiprows=1, unpack=True)
    snaps = snaps.astype(int)
    Lbox_hkpc = 205000. # ckpc/h

elif sim_name == "TNG100":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_tng75.txt"), skiprows=1, unpack=True)
    snaps = snaps.astype(int)
    Lbox_hkpc = 75000. # ckpc/h
    
elif sim_name == "Illustris":
    snaps, _, zs = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_orig.txt"), skiprows=1, unpack=True)
    snaps = snaps.astype(int)
    Lbox_hkpc = 75000. # ckpc/h

elif sim_name == "SIMBA":
    snaps = np.array([125, 134])
    zs = np.array([0.5, 0.3])
    Lbox_hkpc = 100000. # ckpc/h
    
elif sim_name == "MTNG":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_mtng.txt"), skiprows=1, unpack=True)
    snaps = snaps.astype(int)
    type_sim = "_fp"
    Lbox_hkpc = 500000. # ckpc/h 

# select redshift
redshift = float(sys.argv[2]) # 0, 0.25, 0.5, 1.0
a = 1./(1+redshift)
ind = np.argmin(np.abs(redshift - zs))
snapshot = snaps[ind]
snap_str = '_'+str(snapshot)
print("redshift, snapshot", redshift, snapshot)
sys.stdout.flush()

# for the parallelizing
n_jump = n_chunks//n_ranks
assert n_chunks % n_ranks == 0

# whether you want to save all arrays or just some
want_all_dir = False
want_3d = False #False #True

if want_3d:
    if sim_name == "MTNG":
        Ndim = 1620 #1620 #2160 #1080 #1024 # TESTING!!!!!!!!!
    elif sim_name == "Illustris":
        Ndim = 910
    else:
        Ndim = 512
    if "Y_compton" in map_types:
        y_3d = np.zeros((Ndim, Ndim, Ndim), dtype=np.float32)
    if "b" in map_types:
        b_xy_3d = np.zeros((Ndim, Ndim, Ndim), dtype=np.float32)
    if "tau" in map_types:
        tau_3d = np.zeros((Ndim, Ndim, Ndim), dtype=np.float32)
    ncell = Ndim
else:
    # for the histograming
    if sim_name == "Illustris":
        nbins = 2001
    elif sim_name == "SIMBA":
        nbins = 2001
    elif sim_name == "TNG100":
        nbins = 2001
    else:
        nbins = 10001
    ranges = ((0., Lbox_hkpc),(0., Lbox_hkpc))
    nbins2d = (nbins-1, nbins-1)
    nbins2d = np.asarray(nbins2d).astype(np.int64)
    ranges = np.asarray(ranges).astype(np.float64)
    
    # initialize empty maps
    if want_all_dir:
        y_yz = np.zeros((nbins-1, nbins-1))
        y_zx = np.zeros((nbins-1, nbins-1))
        b_yz = np.zeros((nbins-1, nbins-1))
        b_zx = np.zeros((nbins-1, nbins-1))
        tau_yz = np.zeros((nbins-1, nbins-1))
        tau_zx = np.zeros((nbins-1, nbins-1))
    else:
        y_xy = np.zeros((nbins-1, nbins-1))
        b_xy = np.zeros((nbins-1, nbins-1))
        tau_xy = np.zeros((nbins-1, nbins-1))
    ncell = nbins-1
    
# for each chunk, find which halos are part of it
for i in range(myrank*n_jump, (myrank+1)*n_jump):

    # read positions of DM particles
    if sim_name == "TNG300":
        hfile = h5py.File(basePath+f'snapdir_{snapshot:03d}/snap_{snapshot:03d}.{i:d}.hdf5')[PartType]
    elif sim_name == "TNG100":
        hfile = h5py.File(basePath+f'snapdir_{snapshot:03d}/snap_{snapshot:03d}.{i:d}.hdf5')[PartType]
    elif sim_name == "Illustris":
        hfile = h5py.File(basePath+f'snapdir_{snapshot:03d}/snap_{snapshot:03d}.{i:d}.hdf5')[PartType]
    elif sim_name == "CAMELS":
        hfile = h5py.File(basePath+f'snap_{snapshot:03d}.hdf5')[PartType]
    elif sim_name == "SIMBA":
        hfile = h5py.File(basePath+f'snap_m100n1024_{snapshot:03d}.hdf5')[PartType]
    elif sim_name == "MTNG":
        hfile = h5py.File(basePath+f'snapdir_{snapshot:03d}/snapshot_{snapshot:03d}.{i:d}.hdf5')[PartType]
    #print(list(hfile.keys()))

    # select all fields of interest
    C = hfile['Coordinates'][:]
    EA = hfile['ElectronAbundance'][:]
    IE = hfile['InternalEnergy'][:]
    D = hfile['Density'][:]
    M = hfile['Masses'][:]
    V = hfile['Velocities'][:]

    if sim_name == "MTNG":
        C *= 1.e3 # ckpc/h
        D /= 1.e3**3 # Msun/(ckpc/h)^3

    # for each cell, compute its total volume (gas mass by gas density) and convert density units
    dV = M/D # ckpc/h^3 (units don't matter cause we divide)
    D *= unit_dens # g/ccm^3

    # obtain electron temperature, electron number density and velocity
    Te = (gamma - 1.)*IE/k_B * 4*m_p/(1 + 3*X_H + 4*X_H*EA) * unit_c # K
    ne = EA*X_H*D/m_p # ccm^-3
    Ve = V*np.sqrt(a) # km/s
    del EA, IE, M; gc.collect()
    print("mean temperature = ", np.mean(Te))

    # Y, b, tau signal for each voxel
    # netedvunitvol = K; const cm^2/K; Mpc_to_cm = kpc_to_cm*1000.
    Y_chunk = const*(ne*Te*dV)*unit_vol/(a*Lbox_hkpc*(kpc_to_cm/h)/ncell)**2. # unitless
    b_chunk = sigma_T*(ne[:, None]*(Ve/c)*dV[:, None])*unit_vol/(a*Lbox_hkpc*(kpc_to_cm/h)/ncell)**2. # unitless
    tau_chunk = sigma_T*(ne*dV)*unit_vol/(a*Lbox_hkpc*(kpc_to_cm/h)/ncell)**2. # unitless
    print("chunk = ", i, C[:, 0].max(), C[:, 1].max(), C[:, 2].max())
    sys.stdout.flush()
    del Te, ne, Ve, D, V, dV; gc.collect()
    
    # build kdtree
    C %= Lbox_hkpc
    #tree = spatial.cKDTree(C, boxsize=Lbox_hkpc)
    #print("built tree")

    if want_3d:
        if "Y_compton" in map_types:
            numba_tsc_3D(C, y_3d, Lbox_hkpc, weights=Y_chunk)
        if "b" in map_types:
            numba_tsc_3D(C, b_xy_3d, Lbox_hkpc, weights=b_chunk[:, 2])
        if "tau" in map_types:
            numba_tsc_3D(C, tau_3d, Lbox_hkpc, weights=tau_chunk)
    else:
        
        # flatten out in each of three directions (10 times faster than histogramdd
        if want_all_dir:
            y_yz += hist2d_numba_seq(np.array([C[:, 1], C[:, 2]]), bins=nbins2d, ranges=ranges, weights=Y_chunk)
            y_zx += hist2d_numba_seq(np.array([C[:, 2], C[:, 0]]), bins=nbins2d, ranges=ranges, weights=Y_chunk)
        else:
            y_xy += hist2d_numba_seq(np.array([C[:, 0], C[:, 1]]), bins=nbins2d, ranges=ranges, weights=Y_chunk)
        
        # flatten out in each of three directions
        if want_all_dir:
            b_yz += hist2d_numba_seq(np.array([C[:, 1], C[:, 2]]), bins=nbins2d, ranges=ranges, weights=b_chunk[:, 0])
            b_zx += hist2d_numba_seq(np.array([C[:, 2], C[:, 0]]), bins=nbins2d, ranges=ranges, weights=b_chunk[:, 1])
        else:
            b_xy += hist2d_numba_seq(np.array([C[:, 0], C[:, 1]]), bins=nbins2d, ranges=ranges, weights=b_chunk[:, 2])
        
        # flatten out in each of three directions
        if want_all_dir:
            tau_yz += hist2d_numba_seq(np.array([C[:, 1], C[:, 2]]), bins=nbins2d, ranges=ranges, weights=tau_chunk)
            tau_zx += hist2d_numba_seq(np.array([C[:, 2], C[:, 0]]), bins=nbins2d, ranges=ranges, weights=tau_chunk)
        else:
            tau_xy += hist2d_numba_seq(np.array([C[:, 0], C[:, 1]]), bins=nbins2d, ranges=ranges, weights=tau_chunk)

    del C, Y_chunk, b_chunk, tau_chunk; gc.collect()
    
if want_3d:
    if "Y_compton" in map_types:
        np.save(f"{save_dir}/Y_compton_3d_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npy", y_3d)
    if "b" in map_types:
        np.save(f"{save_dir}/b_xy_3d_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npy", b_xy_3d)
    if "tau" in map_types:
        np.save(f"{save_dir}/tau_3d_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npy", tau_3d)
else:
    
    # save tSZ maps
    if want_all_dir:
        np.save(f"{save_dir}/Y_compton_yz_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npy", y_yz)
        np.save(f"{save_dir}/Y_compton_zx_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npy", y_zx)
    else:
        np.save(f"{save_dir}/Y_compton_xy_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npy", y_xy)
    
    # save kSZ maps
    if want_all_dir:
        np.save(f"{save_dir}/b_yz_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npy", b_yz)
        np.save(f"{save_dir}/b_zx_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npy", b_zx)
    else:
        np.save(f"{save_dir}/b_xy_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npy", b_xy)
    
    # save tau maps
    if want_all_dir:
        np.save(f"{save_dir}/tau_yz_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npy", tau_yz)
        np.save(f"{save_dir}/tau_zx_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npy", tau_zx)
    else:
        np.save(f"{save_dir}/tau_xy_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npy", tau_xy)
    
