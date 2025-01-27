"""
full dumps:
51, 69, 80, 94, 129, 151, 179, 214, 237, 264
"""
import sys
import os
import gc
import time
import itertools

import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy import spatial
import h5py

import illustris_python as il
import numba
from numba import njit

from mpi4py import MPI
"""
mpirun -np 32 python calculate_profile_emre_fixed_bug_T.py 32 0.5; mpirun -np 32 python calculate_profile_emre_fixed_bug_T.py 32 0.25; mpirun -np 32 python calculate_profile_emre_fixed_bug_T.py 32 0.0; mpirun -np 32 python calculate_profile_combo_emre_fixed_bug_T.py 32 1.0 # 179 kinda exist (no rank 0)
"""

myrank = MPI.COMM_WORLD.Get_rank() # TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#myrank = 1
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

def concat_to_arr(lists, dtype=np.int64):
    '''Concatenate an iterable of lists to a flat Numpy array.
    Returns the concatenated array and the index where each list starts.
    '''
    starts = np.empty(len(lists) + 1, dtype=np.int64)
    starts[0] = 0
    starts[1:] = np.cumsum(np.fromiter((len(l) for l in lists), count=len(lists), dtype=np.int64))
    N = starts[-1]
    res = np.fromiter(itertools.chain.from_iterable(lists), count=N, dtype=dtype)
    return res, starts

@njit(parallel=True)
def calc_sums(dV, P_chunk, Te, ne, inds_arr, starts):
    N = len(starts)-1
    Vs = np.zeros(N, dtype=np.float32)
    Ps = np.zeros(N, dtype=np.float32)
    Ts = np.zeros(N, dtype=np.float32)
    ns = np.zeros(N, dtype=np.float32)
    Ns = np.zeros(N, dtype=np.float32)
    for p in numba.prange(N):
        Vs[p] = np.sum(dV[inds_arr[starts[p]:starts[p+1]]])
        Ps[p] = np.sum(P_chunk[inds_arr[starts[p]:starts[p+1]]])
        Ts[p] = np.sum(Te[inds_arr[starts[p]:starts[p+1]]])
        ns[p] = np.sum(ne[inds_arr[starts[p]:starts[p+1]]])
        Ns[p] = starts[p+1] - starts[p]
    return Vs, Ps, Ts, ns, Ns

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
    save_dir = "/freya/ptmp/mpa/boryanah/data_sz/emre/" # virgo #TESTING!!!!!!!!!!!!!
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
#rbins = np.logspace(-2, 1.7, 26) # ratio
#rbins = np.logspace(1., 1.75, 6) # ratio
#rbins = np.logspace(1., 1.6, 3) # ratio
rbins = np.logspace(-2, 1., 21) # ratio og

# initialize arrays for each mass bin and each r bins
P_e = np.zeros((len(i_sort), len(rbins)-1), dtype=np.float32)
n_e = np.zeros((len(i_sort), len(rbins)-1), dtype=np.float32)
m_g = np.zeros((len(i_sort), len(rbins)-1), dtype=np.float32)
T_e = np.zeros((len(i_sort), len(rbins)-1), dtype=np.float32)
N_v = np.zeros((len(i_sort), len(rbins)-1), dtype=np.float32)
V_d = np.zeros((len(i_sort), len(rbins)-1), dtype=np.float32)

# for the parallelizing
n_jump = n_chunks//n_ranks
assert n_chunks % n_ranks == 0

# randomize it because the tree building takes too long
inds_chunks = np.arange(n_chunks)
#np.random.shuffle(inds_chunks) # I think we didn't have it before
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

    # select all fields of interest (float32)
    C = hfile['Coordinates'][:]
    EA = hfile['ElectronAbundance'][:]
    IE = hfile['InternalEnergy'][:]
    D = hfile['Density'][:]
    M = hfile['Masses'][:]
    #V = hfile['Velocities'][:]

    if sim_name == "MTNG":
        C *= 1.e3 # ckpc/h
        D /= 1.e3**3 # Msun/(ckpc/h)^3
    
    # for each cell, compute its total volume (gas mass by gas density) and convert density units
    dV = M/D # ckpc/h^3 (units don't matter cause we divide)
    D *= unit_dens # g/ccm^3

    # obtain electron temperature, electron number density and velocity
    Te = (gamma - 1.)*IE/k_B * 4*m_p/(1 + 3*X_H + 4*X_H*EA) * unit_c # K
    ne = EA*X_H*D/m_p # ccm^-3
    #Ve = V*np.sqrt(a) # km/s
    del EA, IE; gc.collect()
    print("mean temperature, redshift = ", np.mean(Te), redshift)
        
    # pressure of each voxel
    P_chunk = ne*k_B*Te # erg/ccm^3
    print("chunk = ", i, ind_chunk)

    # build kdtree
    C %= Lbox_hkpc
    t = time.time()
    tree = spatial.cKDTree(C, boxsize=Lbox_hkpc)
    print("built tree in", time.time()-t)
    sys.stdout.flush()
    
    t = time.time()
    # loop over all halos of interest
    for j in range(len(i_sort)):
        # relevant halo quantities
        pos = poss[j] # ckpc/h
        r200c = r200cs[j] # ckpc/h
        m200c = m200cs[j]/h # Msun
        vbins = 4/3.*np.pi*(rbins*r200c)**3 # (ckpc/h)^3
        dvols = vbins[1:]-vbins[:-1]

        # mass bin and radial bin
        for k in range(-1, len(rbins)-1):
            inds = np.asarray(tree.query_ball_point(pos, r200c*rbins[k+1]), dtype=np.int64).flatten()
            if len(inds) > 0:
                V_outer = np.sum(dV[inds])
                P_outer = np.sum(P_chunk[inds]*dV[inds])
                T_outer = np.sum(Te[inds]*M[inds])
                n_outer = np.sum(ne[inds]*dV[inds])
                m_outer = np.sum(M[inds])
                
                if k >= 0:
                    V_d[j, k] += (V_outer - V_inner)
                    P_e[j, k] += (P_outer - P_inner)
                    T_e[j, k] += (T_outer - T_inner)
                    n_e[j, k] += (n_outer - n_inner)
                    m_g[j, k] += (m_outer - m_inner)
                    N_v[j, k] += len(inds) - len(inds_inner)

                V_inner = V_outer
                P_inner = P_outer
                T_inner = T_outer
                n_inner = n_outer
                m_inner = m_outer
                inds_inner = inds
            if len(inds) == 0 and k == -1:
                P_inner = 0.
                T_inner = 0.
                n_inner = 0.
                m_inner = 0.
                V_inner = 0.
                inds_inner = np.array([])
    print("looped over halos in", time.time()-t)
    
# save fields
np.savez(f"{save_dir}/prof_sph_r200c_m200c_{mmin}Msun_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npz", rbins=rbins, P_e=P_e, T_e=T_e, n_e=n_e, m_g=m_g,  N_v=N_v, V_d=V_d, inds_halo=i_sort)
