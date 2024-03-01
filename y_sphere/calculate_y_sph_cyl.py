"""
full dumps:
51, 69, 80, 94, 129, 151, 179, 214, 237, 264
"""
import sys
import os
import gc
import time

import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy import spatial
import h5py

import illustris_python as il

from mpi4py import MPI

"""
mpirun -np 64 python calculate_y_sph_cyl.py 64 1.0

mpirun -np 8 python calculate_y_sph_cyl.py 8 1.0 yz_zx

mpirun -np 16 python calculate_y_sph_cyl.py 16 1.0
mpirun -np 16 python calculate_y_sph_cyl.py 16 0.5
mpirun -np 16 python calculate_y_sph_cyl.py 16 0.25
mpirun -np 16 python calculate_y_sph_cyl.py 16 0.0
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

# sim params
h = 67.74/100.
Omega_m = 0.3089
unit_mass = 1.e10*(solar_mass/h) # g
unit_dens = 1.e10*(solar_mass/h)/(kpc_to_cm/h)**3 # g/cm**3 # note density has units of h in it
unit_vol = (kpc_to_cm/h)**3 # cm^3 cancels unit dens division from previous, so h doesn't matter neither does kpc

# simulation info
sim_name = "MTNG"
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
elif sim_name == "MTNG":
    basePath = "/virgotng/mpa/MTNG/Hydro-Arepo/MTNG-L500-4320-A/output/"
    n_chunks = 640
    save_dir = "/freya/ptmp/mpa/boryanah/data_sz/" # virgo
PartType = 'PartType0' # gas cells
os.makedirs(save_dir, exist_ok=True)

# simulation choices
if sim_name == "TNG300":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_tng205.txt"), skiprows=1, unpack=True)
    snaps = snaps.astype(int)

    data_dir = "/n/home13/bhadzhiyska/TNG_transfer/data_2500/"
    Lbox_hkpc = 205000. # ckpc/h
    
elif sim_name == "MTNG":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_mtng.txt"), skiprows=1, unpack=True)
    snaps = snaps.astype(int)

    type_sim = "_fp"
    data_dir = "/freya/ptmp/mpa/boryanah/data_fp/"
    Lbox_hkpc = 500000. # ckpc/h 

# select redshift
redshift = float(sys.argv[2]) # 0, 0.25, 0.5, 1.0
a = 1./(1+redshift)
ind = np.argmin(np.abs(redshift - zs))
snapshot = snaps[ind]
snap_str = '_'+str(snapshot)
print("redshift, snapshot", redshift, snapshot)

# select orientation
if len(sys.argv) > 3:
    orientation = sys.argv[3]
    orient_str = f"_{orientation}"
    orientations = orientation.split('_')
else:
    orientation = 'xy'
    orient_str = ""
    orientations = ['xy']
    
# load stuff
if sim_name == "TNG300":
    Group_R_Crit200 = np.load(data_dir+"Group_R_Crit200"+snap_str+"_fp.npy") #ckpc/h
    Group_M_Crit200 = np.load(data_dir+"Group_M_Crit200"+snap_str+"_fp.npy")*1.e10 # Msun/h
    GroupPos = np.load(data_dir+"GroupPos"+snap_str+"_fp.npy") #ckpc/h
elif sim_name == "MTNG":
    fields = ['Group_R_Crit200', 'Group_M_Crit200', 'GroupPos']
    try:
        Group_R_Crit200 = np.load(data_dir+f"Group_R_Crit200{type_sim}{snap_str}.npy")*1000. #ckpc/h
        Group_M_Crit200 = np.load(data_dir+f"Group_M_Crit200{type_sim}{snap_str}.npy")*1.e10 #Msun/h
        GroupPos = np.load(data_dir+f"GroupPos{type_sim}{snap_str}.npy")*1000. #ckpc/h
    except:
        for field in fields:
            groups = il.groupcat.loadHalos(basePath, snapshot, fields=[field])
            np.save(data_dir+"/"+field+type_sim+"_"+str(snapshot)+".npy", groups)
        Group_R_Crit200 = np.load(data_dir+f"Group_R_Crit200{type_sim}{snap_str}.npy")*1000. #ckpc/h
        Group_M_Crit200 = np.load(data_dir+f"Group_M_Crit200{type_sim}{snap_str}.npy")*1.e10 #Msun/h
        GroupPos = np.load(data_dir+f"GroupPos{type_sim}{snap_str}.npy")*1000. #ckpc/h

# select halos    
#i_sort = np.arange(N_gal)
mmin = '1e12' # Msun
N_gal = np.sum(Group_M_Crit200 > float(mmin)*h)
i_sort = np.argsort(Group_M_Crit200)[::-1]
i_sort = i_sort[:N_gal]
poss = GroupPos[i_sort]
r200cs = Group_R_Crit200[i_sort]
m200cs = Group_M_Crit200[i_sort]
print("N_gal", N_gal)
del Group_R_Crit200, Group_M_Crit200, GroupPos
gc.collect()
print(m200cs[-10:], np.sum(m200cs == 0.), r200cs[np.argmin(m200cs)], np.argmin(m200cs))
print(f"lowest mass {np.min(m200cs):.3e}")
sys.stdout.flush()

# for the parallelizing
n_jump = n_chunks//n_ranks
assert n_chunks % n_ranks == 0

# initialize
Y_200c_sph = np.zeros(N_gal)
Y_200c_cyl_xy = np.zeros(N_gal)
Y_200c_cyl_yz = np.zeros(N_gal)
Y_200c_cyl_zx = np.zeros(N_gal)
b_200c_sph = np.zeros((N_gal, 3))
b_200c_cyl_xy = np.zeros(N_gal)
b_200c_cyl_yz = np.zeros(N_gal)
b_200c_cyl_zx = np.zeros(N_gal)
tau_200c_sph = np.zeros(N_gal)
tau_200c_cyl_xy = np.zeros(N_gal)
tau_200c_cyl_yz = np.zeros(N_gal)
tau_200c_cyl_zx = np.zeros(N_gal)

np.random.seed(300)

# randomize it because the tree building takes too long
inds_chunks = np.arange(n_chunks)
np.random.shuffle(inds_chunks)
inds_rank = inds_chunks[myrank*n_jump: (myrank+1)*n_jump]

# for each chunk, find which halos are part of it
for i in range(len(inds_rank)):
    ind_chunk = inds_rank[i]

    # read positions of DM particles
    if sim_name == "TNG300":
        hfile = h5py.File(basePath+f'snapdir_{snapshot:03d}/snap_{snapshot:03d}.{ind_chunk:d}.hdf5')[PartType]
    elif sim_name == "CAMELS":
        hfile = h5py.File(basePath+f'snap_{snapshot:03d}.hdf5')[PartType]
    elif sim_name == "MTNG":
        hfile = h5py.File(basePath+f'snapdir_{snapshot:03d}/snapshot_{snapshot:03d}.{ind_chunk:d}.hdf5')[PartType]
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
    Y_chunk = const*(ne*Te*dV)*unit_vol/(kpc_to_cm*1000.)**2. # Mpc^2
    b_chunk = sigma_T*(ne[:, None]*(Ve/c)*dV[:, None])*unit_vol/(kpc_to_cm*1000.)**2. # Mpc^2 (note V/c in km/cm)
    tau_chunk = sigma_T*(ne*dV)*unit_vol/(kpc_to_cm*1000.)**2. # Mpc^2    
    print("chunk = ", i, ind_chunk, C[:, 0].max(), C[:, 1].max(), C[:, 2].max())
    sys.stdout.flush()
    
    # build kdtree
    if orient_str == "":
        t1 = time.time()
        C %= Lbox_hkpc
        tree = spatial.cKDTree(C, boxsize=Lbox_hkpc)
        print("built tree", time.time()-t1)

    # build kdtree
    if "xy" in orientations:
        t1 = time.time()
        C_xy = np.vstack((C[:, 0], C[:, 1])).T # x and y
        C_xy %= Lbox_hkpc
        tree_xy = spatial.cKDTree(C_xy, boxsize=Lbox_hkpc)
        print("built xy tree", time.time()-t1)

    # build kdtree
    if "yz" in orientations:
        t1 = time.time()
        C_yz = np.vstack((C[:, 1], C[:, 2])).T # y and z
        C_yz %= Lbox_hkpc
        tree_yz = spatial.cKDTree(C_yz, boxsize=Lbox_hkpc)
        print("built yz tree", time.time()-t1)

    # build kdtree
    if "zx" in orientations:
        t1 = time.time()
        C_zx = np.vstack((C[:, 2], C[:, 0])).T # z and x
        C_zx %= Lbox_hkpc
        tree_zx = spatial.cKDTree(C_zx, boxsize=Lbox_hkpc)
        print("built zx tree", time.time()-t1)
    sys.stdout.flush()
    
    for j in range(len(i_sort)):

        # relevant halo quantities
        pos = poss[j] # ckpc/h
        pos_xy = np.array([pos[0], pos[1]]) # x and y
        pos_yz = np.array([pos[1], pos[2]]) # y and z
        pos_zx = np.array([pos[2], pos[0]]) # z and x
        r200c = r200cs[j] # ckpc/h

        if orient_str == "":
            # check whether pos, r200c is within coords 
            inds = np.asarray(tree.query_ball_point(pos, r200c), dtype=np.int).flatten()
            if len(inds) > 0:
                # if so, add to the Y signal of that object
                Y_200c_sph[j] += np.sum(Y_chunk[inds])
                b_200c_sph[j, 0] += np.sum(b_chunk[inds, 0])
                b_200c_sph[j, 1] += np.sum(b_chunk[inds, 1])
                b_200c_sph[j, 2] += np.sum(b_chunk[inds, 2])
                tau_200c_sph[j] += np.sum(tau_chunk[inds])
            del inds; gc.collect()
            
        if "xy" in orientations:
            # check whether pos, r200c is within coords 
            inds = np.asarray(tree_xy.query_ball_point(pos_xy, r200c), dtype=np.int).flatten()
            if len(inds) > 0:
                # if so, add to the Y signal of that object
                Y_200c_cyl_xy[j] += np.sum(Y_chunk[inds])
                b_200c_cyl_xy[j] += np.sum(b_chunk[inds, 2])
                tau_200c_cyl_xy[j] += np.sum(tau_chunk[inds])
            del inds; gc.collect()
            
        if "yz" in orientations:
            # check whether pos, r200c is within coords 
            inds = np.asarray(tree_yz.query_ball_point(pos_yz, r200c), dtype=np.int).flatten()
            if len(inds) > 0:
                # if so, add to the Y signal of that object
                Y_200c_cyl_yz[j] += np.sum(Y_chunk[inds])
                b_200c_cyl_yz[j] += np.sum(b_chunk[inds, 0])
                tau_200c_cyl_yz[j] += np.sum(tau_chunk[inds])
            del inds; gc.collect()
            
        if "zx" in orientations:
            # check whether pos, r200c is within coords 
            inds = np.asarray(tree_zx.query_ball_point(pos_zx, r200c), dtype=np.int).flatten()
            if len(inds) > 0:
                # if so, add to the Y signal of that object
                Y_200c_cyl_zx[j] += np.sum(Y_chunk[inds])
                b_200c_cyl_zx[j] += np.sum(b_chunk[inds, 1])
                tau_200c_cyl_zx[j] += np.sum(tau_chunk[inds])
            del inds; gc.collect()
            
    sys.stdout.flush()

np.savez(f"{save_dir}/SZ{orient_str}_sph_cyl_r200c_m200c_{mmin}Msun_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npz", Y_200c_sph=Y_200c_sph, Y_200c_cyl_xy=Y_200c_cyl_xy, Y_200c_cyl_yz=Y_200c_cyl_yz, Y_200c_cyl_zx=Y_200c_cyl_zx, b_200c_sph=b_200c_sph, b_200c_cyl_xy=b_200c_cyl_xy, b_200c_cyl_yz=b_200c_cyl_yz, b_200c_cyl_zx=b_200c_cyl_zx, tau_200c_sph=tau_200c_sph, tau_200c_cyl_xy=tau_200c_cyl_xy, tau_200c_cyl_yz=tau_200c_cyl_yz, tau_200c_cyl_zx=tau_200c_cyl_zx, inds_halo=i_sort)
