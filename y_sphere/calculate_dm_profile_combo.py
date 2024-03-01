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

import illustris_python as il

from mpi4py import MPI
"""
cd /u/boryanah/repos/SZ_TNG/y_sphere
mpirun -np 32 python calculate_dm_profile_combo.py 32 1.0 fp
mpirun -np 32 python calculate_dm_profile_combo.py 32 0.5 fp
mpirun -np 32 python calculate_dm_profile_combo.py 32 0.25 fp
mpirun -np 32 python calculate_dm_profile_combo.py 32 0.0 fp

cd /u/boryanah/repos/SZ_TNG/y_sphere
mpirun -np 32 python calculate_dm_profile_combo.py 32 1.0 dm
mpirun -np 32 python calculate_dm_profile_combo.py 32 0.5 dm
mpirun -np 32 python calculate_dm_profile_combo.py 32 0.25 dm
mpirun -np 32 python calculate_dm_profile_combo.py 32 0.0 dm
"""

myrank = MPI.COMM_WORLD.Get_rank()
n_ranks = int(sys.argv[1])
print("myrank", myrank)

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

# particle mass
mpart = 1.3293e+08 # total mass (dark+baryon) Msun/h
mpart /= h # Msun
mpart *= solar_mass # g

# select which simulation
type_sim = f"_{sys.argv[3]}" # "_fp", "_dm"

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
    if type_sim == "_fp":
        basePath = "/virgotng/mpa/MTNG/Hydro-Arepo/MTNG-L500-4320-A/output/"
    elif type_sim == "_dm":
        basePath = "/virgotng/mpa/MTNG/DM-Arepo/MTNG-L500-4320-A/output/"
    n_chunks = 640
    save_dir = "/freya/ptmp/mpa/boryanah/data_sz/" # virgo
PartType = 'PartType1' # dm cells
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

    if "_dm" == type_sim:
        data_dir = f"/freya/ptmp/mpa/boryanah/dm_arepo/data{type_sim}/"
    else:
        data_dir = f"/freya/ptmp/mpa/boryanah/data{type_sim}/"
    Lbox_hkpc = 500000. # cMpc/h 

# select redshift
redshift = float(sys.argv[2]) # 0, 0.25, 0.5, 1.0
ind = np.argmin(np.abs(redshift - zs))
snapshot = snaps[ind]
snap_str = '_'+str(snapshot)
print("redshift, snapshot", redshift, snapshot)

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

# bins for getting profiles
rbins = np.logspace(-2, 1, 21) # ratio

# initialize arrays for each mass bin and each r bins
V_d = np.zeros((len(i_sort), len(rbins)-1))
N_v = np.zeros((len(i_sort), len(rbins)-1))
rho_dm = np.zeros((len(i_sort), len(rbins)-1))

# for the parallelizing
n_jump = n_chunks//n_ranks
assert n_chunks % n_ranks == 0

# for each chunk, find which halos are part of it
for i in range(myrank*n_jump, (myrank+1)*n_jump):
    print("chunk", i)
    
    # read positions of DM particles
    if sim_name == "TNG300":
        hfile = h5py.File(basePath+f'snapdir_{snapshot:03d}/snap_{snapshot:03d}.{i:d}.hdf5')[PartType]
    elif sim_name == "CAMELS":
        hfile = h5py.File(basePath+f'snap_{snapshot:03d}.hdf5')[PartType]
    elif sim_name == "MTNG":
        hfile = h5py.File(basePath+f'snapdir_{snapshot:03d}/snapshot_{snapshot:03d}.{i:d}.hdf5')[PartType]

    # select all fields of interest
    C = hfile['Coordinates'][:]
    if sim_name == "MTNG":
        C *= 1.e3 # ckpc/h

    # build kdtree
    C %= Lbox_hkpc
    tree = spatial.cKDTree(C, boxsize=Lbox_hkpc)
    print("built tree")
    sys.stdout.flush()

    # loop over all halos of interest
    for j in range(len(i_sort)):
        # relevant halo quantities
        pos = poss[j] # ckpc/h
        r200c = r200cs[j] # ckpc/h

        # compute volume in shells
        vbins = 4/3.*np.pi*(rbins*r200c)**3 # (ckpc/h)^3
        dvols = vbins[1:]-vbins[:-1]
        dvols *= unit_vol # ccm^3 
        
        # mass bin and radial bin
        for k in range(-1, len(rbins)-1):
            inds = np.asarray(tree.query_ball_point(pos, r200c*rbins[k+1]), dtype=np.int).flatten()
            if len(inds) > 0:                
                rho_outer = len(inds)*mpart # g

                if k >= 0:
                    V_d[j, k] += dvols[k] # ccm^3
                    rho_dm[j, k] += (rho_outer - rho_inner)
                    N_v[j, k] += 1#len(inds) - len(inds_inner)
                    
                rho_inner = rho_outer
                inds_inner = inds
            if len(inds) == 0 and k == -1:
                inds_inner = np.array([])
                rho_inner = 0.

# save fields
np.savez(f"{save_dir}/prof_dm{type_sim}_sph_r200c_m200c_{mmin}Msun_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npz", rbins=rbins, rho_dm=rho_dm, N_v=N_v, V_d=V_d, inds_halo=i_sort)
