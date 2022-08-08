import sys
import os

import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy import spatial

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
Lbox_kpc = 205000. # ckpc/h
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
snapshot = int(sys.argv[1]) # 78, 67, 59, 50 # 91 # 99
redshift = zs[snaps == snapshot]
snap_str = '_'+str(snapshot)
#if snapshot == 99:
#    snap_str = ''

# angular distance 
cosmo = FlatLambdaCDM(H0=h*100., Om0=Omega_m, Tcmb0=T_CMB)
d_L = cosmo.luminosity_distance(redshift).to(u.kpc).value # kpc
d_A = d_L/(1.+redshift)**2 # dA = dL/(1+z)^2 # kpc
d_A *= kpc_to_cm # cm
print("d_A = ", d_A)

# select halos of interest
data_dir = "/n/home13/bhadzhiyska/TNG_transfer/data_2500/"
save_dir = "/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/SZ_TNG/"
Group_R_Crit500 = np.load(data_dir+"Group_R_Crit500"+snap_str+"_fp.npy") #ckpc/h
Group_M_Crit500 = np.load(data_dir+"Group_M_Crit500"+snap_str+"_fp.npy")*1.e10 # Msun/h
GroupPos = np.load(data_dir+"GroupPos"+snap_str+"_fp.npy") #ckpc/h
N_gal = 50000
i_sort = (np.argsort(Group_M_Crit500)[::-1])[:N_gal]
poss = GroupPos[i_sort]
r500cs = Group_R_Crit500[i_sort]
m500cs = Group_M_Crit500[i_sort]

# bins for getting profiles
rbins = np.logspace(1, 4, 41) # ckpc
vbins = 4/3.*np.pi*rbins**3
dvol = vbins[1:]-vbins[:-1]
rbin = (rbins[1:]+rbins[:-1])*.5

dY = np.zeros(len(i_sort))
# for each chunk, find which halos are part of it
for i in range(n_chunks):

    # read saved fields of interest
    C = np.load(f"{save_dir}/position_chunk_{i:d}_snap_{snapshot:d}.npy")%Lbox_kpc # cMpc/h (MTNG) and ckpc/h (TNG)
    Te = np.load(f"{save_dir}/temperature_chunk_{i:d}_snap_{snapshot:d}.npy") # K
    ne = np.load(f"{save_dir}/number_density_chunk_{i:d}_snap_{snapshot:d}.npy") # comoving cm^-3 # True for TNG and mixed for MTNG because of unit difference
    dV = np.load(f"{save_dir}/volume_chunk_{i:d}_snap_{snapshot:d}.npy") # cMpc/h^3 (MTNG) and ckpc/h^3 (TNG)
    #Ve = np.load(f"{save_dir}/velocity_chunk_{i:d}_snap_{snapshot:d}.npy")*1.e5 # cm/s from km/s

    # Y signal for each voxel
    Y_chunk = const*(ne*Te*dV)*unit_vol/(kpc_to_cm*1000.)**2. #/d_A**2 # Mpc^2 
    #P = ne*k_B*Te
    print("chunk = ", i, C[:, 0].max(), C[:, 1].max(), C[:, 2].max())

    # build kdtree
    tree = spatial.cKDTree(C, boxsize=Lbox_kpc)
    print("built tree")

    for j in range(len(i_sort)):

        # relevant halo quantities
        pos = poss[j] # ckpc
        r500c = r500cs[j] # ckpc
        #vol = 4./3*np.pi*(r500c*kpc_to_cm)**3 # comoving cm^3

        # check whether pos, r500c is within coords 
        inds = np.asarray(tree.query_ball_point(pos, r500c), dtype=np.int).flatten()
        #print(i_sort[j], r500c, pos, len(inds), C[inds[-10:]])
        if len(inds) == 0: continue
        signal = np.sum(Y_chunk[inds])
        print("exists!", j, signal)

        # if so, add to the Y signal of that object
        dY[j] += signal


np.savez(f"{save_dir}/SZ_500c_nr_{N_gal:d}_snap_{snapshot:d}.npz", Y_500c=dY, halo_inds=i_sort, M_500c=m500cs)
if False:
    dist2 = (np.sum((C-pos)**2, axis=1))
    choice = dist2 < r500c**2

    print(np.sum(dV[choice]))
    print(vol)
    #Y = 1./d_A**2*const*np.mean((ne*Te)[choice])*vol
    Y = 1./d_A**2*const*np.sum((ne*Te*dV)[choice])*unit_vol # unit vol changes dV to cm^3
    print(Y)

    P_r = np.zeros(len(rbin))
    for j in range(len(rbin)):
        choice = (rbins[j]**2 < dist2) & (rbins[j+1]**2 >= dist2)
        P_r[j] = np.sum((P*(dV/h**3))[choice])/dvol[j]
    np.save(f"data/press_hi{halo_index:04d}_snap{snapshot:d}.npy", P_r)
    print("P_r = ", P_r)
    np.save(f"data/rbin.npy", rbin)
