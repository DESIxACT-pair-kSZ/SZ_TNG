import sys
import os
import gc

import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy import spatial

from mpi4py import MPI
# mpirun -np 16 python calculate_y_sph.py 16 67; mpirun -np 16 python calculate_y_sph.py 16 78
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
T_CMB = 2.7255 # K
Lbox_hkpc = 205000. # ckpc/h
unit_mass = 1.e10*(solar_mass/h) # g
unit_dens = 1.e10*(solar_mass/h)/(kpc_to_cm/h)**3 # g/cm**3 # note density has units of h in it
unit_vol = (kpc_to_cm/h)**3 # cm^3 cancels unit dens division from previous, so h doesn't matter neither does kpc

# simulation choices
#sim_name = "MTNG"
sim_name = "TNG300"
if sim_name == "TNG300":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_tng205.txt"), skiprows=1, unpack=True)
elif sim_name == "MNTG":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_mtng.txt"), skiprows=1, unpack=True)
snaps = snaps.astype(int)
# 99, 91, 84, 78, 72, 67, 63, 59, 56, 53, 50
# 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.79, 0.89, 1.
n_chunks = 600
snapshot = int(sys.argv[2]) # 78, 67, 59, 50 # 91 # 99
redshift = zs[snaps == snapshot]
snap_str = '_'+str(snapshot)

# angular distance 
cosmo = FlatLambdaCDM(H0=h*100., Om0=Omega_m, Tcmb0=T_CMB)
d_L = cosmo.luminosity_distance(redshift).to(u.kpc).value # kpc
d_A = d_L/(1.+redshift)**2 # dA = dL/(1+z)^2 # kpc
d_A *= kpc_to_cm # cm
print("d_A = ", d_A)
sys.stdout.flush()

# select halos of interest
data_dir = "/n/home13/bhadzhiyska/TNG_transfer/data_2500/"
save_dir = "/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/SZ_TNG/"
Group_R_Crit500 = np.load(data_dir+"Group_R_Crit500"+snap_str+"_fp.npy") #ckpc/h
Group_R_TopHat200 = np.load(data_dir+"Group_R_TopHat200"+snap_str+"_fp.npy") #ckpc/h
Group_R_Crit200 = np.load(data_dir+"Group_R_Crit200"+snap_str+"_fp.npy") #ckpc/h
Group_R_Mean200 = np.load(data_dir+"Group_R_Mean200"+snap_str+"_fp.npy") #ckpc/h
Group_M_Crit500 = np.load(data_dir+"Group_M_Crit500"+snap_str+"_fp.npy")*1.e10 # Msun/h
Group_M_TopHat200 = np.load(data_dir+"Group_M_TopHat200"+snap_str+"_fp.npy")*1.e10 # Msun/h
Group_M_Crit200 = np.load(data_dir+"Group_M_Crit200"+snap_str+"_fp.npy")*1.e10 # Msun/h
Group_M_Mean200 = np.load(data_dir+"Group_M_Mean200"+snap_str+"_fp.npy")*1.e10 # Msun/h
GroupPos = np.load(data_dir+"GroupPos"+snap_str+"_fp.npy") #ckpc/h
N_gal = 100000

i_sort = np.arange(N_gal)
#i_sort = (np.argsort(Group_M_Crit500)[::-1])[:N_gal]
#i_sort = (np.argsort(Group_M_TopHat200)[::-1])[:N_gal]
poss = GroupPos[i_sort]
r500cs = Group_R_Crit500[i_sort]
r200ts = Group_R_TopHat200[i_sort]
r200cs = Group_R_Crit200[i_sort]
r200ms = Group_R_Mean200[i_sort]
m500cs = Group_M_Crit500[i_sort]
m200ts = Group_M_TopHat200[i_sort]
m200ms = Group_M_Mean200[i_sort]
m200cs = Group_M_Crit200[i_sort]
del Group_R_Crit500, Group_R_TopHat200, Group_R_Crit200, Group_R_Mean200
del Group_M_Crit500, Group_M_TopHat200, Group_M_Crit200, Group_M_Mean200, GroupPos; gc.collect()

# bins for getting profiles
rbins = np.logspace(-3, 1, 24) # ratio

# mass bins
mbins = np.logspace(12, 14.6, 23) # Msun
mbinc = (mbins[1:]+mbins[:-1])*.5

# initialize arrays for each mass bin and each r bins
# used to be len(mbins)-1
P_e = np.zeros((len(i_sort), len(rbins)-1))
n_e = np.zeros((len(i_sort), len(rbins)-1))
T_e = np.zeros((len(i_sort), len(rbins)-1))
N_v = np.zeros((len(i_sort), len(rbins)-1))
V_d = np.zeros((len(i_sort), len(rbins)-1))
rho_dm = np.zeros((len(i_sort), len(rbins)-1))

# for the parallelizing
n_jump = n_chunks//n_ranks
assert n_chunks % n_ranks == 0

# for each chunk, find which halos are part of it
for i in range(myrank*n_jump, (myrank+1)*n_jump):

    # read saved fields of interest
    C = np.load(f"{save_dir}/position_chunk_{i:d}_snap_{snapshot:d}.npy")%Lbox_hkpc # cMpc/h (MTNG) and ckpc/h (TNG)
    Te = np.load(f"{save_dir}/temperature_chunk_{i:d}_snap_{snapshot:d}.npy") # K
    ne = np.load(f"{save_dir}/number_density_chunk_{i:d}_snap_{snapshot:d}.npy") # comoving cm^-3 # True for TNG and mixed for MTNG because of unit difference
    dV = np.load(f"{save_dir}/volume_chunk_{i:d}_snap_{snapshot:d}.npy") # cMpc/h^3 (MTNG) and ckpc/h^3 (TNG)
    #Ve = np.load(f"{save_dir}/velocity_chunk_{i:d}_snap_{snapshot:d}.npy")*1.e5 # cm/s from km/s
    rho = np.load(f"{save_dir}/density_chunk_{i:d}_snap_{snapshot:d}.npy") # g/ccm^3

    # Y signal for each voxel
    #Y_chunk = const*(ne*Te*dV)*unit_vol/(kpc_to_cm*1000.)**2. #/d_A**2 # Mpc^2 
    P_chunk = ne*k_B*Te # erg/ccm^3
    print("chunk = ", i)

    # build kdtree
    tree = spatial.cKDTree(C, boxsize=Lbox_hkpc)
    print("built tree")

    # loop over all halos of interest
    for j in range(len(i_sort)):

        # relevant halo quantities
        pos = poss[j] # ckpc/h
        r200c = r200cs[j] # ckpc/h
        m200c = m200cs[j]/h # Msun
        vbins = 4/3.*np.pi*(rbins*r200c)**3 # (ckpc/h)^3
        dvols = vbins[1:]-vbins[:-1]

        # mass bin and radial bin
        P_inner = 0.
        T_inner = 0.
        n_inner = 0.
        V_inner = 0.
        rho_inner = 0.
        inds_inner = np.array([])
        for k in range(len(rbins)-1):
            inds = np.asarray(tree.query_ball_point(pos, r200c*rbins[k+1]), dtype=np.int).flatten()
            if len(inds) > 0:
                V_outer = np.sum(dV[inds])
                P_outer = np.sum(P_chunk[inds]*dV[inds])
                T_outer = np.sum(Te[inds])
                n_outer = np.sum(ne[inds]*dV[inds])
                rho_outer = np.sum(rho[inds]*dV[inds])

                V_d[j, k] += (V_outer - V_inner)
                P_e[j, k] += (P_outer - P_inner)
                T_e[j, k] += (T_outer - T_inner)
                n_e[j, k] += (n_outer - n_inner)
                rho_dm[j, k] += (rho_outer - rho_inner)
                N_v[j, k] += len(inds) - len(inds_inner)

                V_inner = V_outer
                P_inner = P_outer
                T_inner = T_outer
                n_inner = n_outer
                rho_inner = rho_outer
                inds_inner = inds

    sys.stdout.flush()

# convert to physical 
V_d /= (1.+redshift)**3.

# save fields (notice M_200c and R_200c in Msun/h and ckpc/h units)
np.savez(f"{save_dir}/prof_sph_m200c_{N_gal:d}_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npz", mbins=mbins, rbins=rbins, P_e=P_e, T_e=T_e, n_e=n_e, rho_dm=rho_dm, N_v=N_v, V_d=V_d, M_200c=m200cs, R_200c=r200cs)
