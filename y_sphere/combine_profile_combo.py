import sys
import os

import numpy as np

"""
python3 combine_profile_combo.py 16 264
python3 combine_profile_combo.py 16 237
python3 combine_profile_combo.py 16 214
python3 combine_profile_combo.py 16 179
"""

#TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! # can get rid of this since you shouldn't use this but random file
extra_str = "_rand"

n_ranks = int(sys.argv[1]) # 16, 32
snapshot = int(sys.argv[2]) # 179, 214, 237, 264

# location stored
save_dir = "/freya/ptmp/mpa/boryanah/data_sz/"
style_str = "prof_sph_r200c_m200c_1e12Msun"

# read the data
data = np.load(f"{save_dir}/{style_str}_snap_{snapshot:d}_rank0_{n_ranks:d}.npz")
inds_halo = data['inds_halo']
N_gal = len(inds_halo)
rbins = data['rbins']
rand_rbins_hckpc = data['rand_rbins_hckpc']
print("read the first one")

# initialize
P_e = np.zeros((N_gal, len(rbins)-1)) # mean electron pressure in each radial and mass bin
n_e = np.zeros((N_gal, len(rbins)-1)) # mean electron number density in each radial and mass bin
T_e = np.zeros((N_gal, len(rbins)-1)) # mean temperature in each radial and mass bin
V_d = np.zeros((N_gal, len(rbins)-1))
N_v = np.zeros((N_gal, len(rbins)-1)) 
N_rand = 5000
rand_V_d = np.zeros((N_rand, len(rand_rbins_hckpc)-1))
rand_P_e = np.zeros((N_rand, len(rand_rbins_hckpc)-1))
rand_n_e = np.zeros((N_rand, len(rand_rbins_hckpc)-1))

for myrank in range(n_ranks):
    print("myrank", myrank)
    data = np.load(f"{save_dir}/{style_str}_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npz")

    P_e += data['P_e']
    n_e += data['n_e']
    T_e += data['T_e']
    V_d += data['V_d']
    N_v += data['N_v']

    rand_V_d += data['V_d']
    rand_P_e += data['P_e']
    rand_n_e += data['n_e']

print("trying to save")
if "rand" in extra_str:
    np.savez(f"{save_dir}/{style_str}{extra_str}_snap_{snapshot:d}.npz", rand_rbins_hckpc=rand_rbins_hckpc, rand_P_e=rand_P_e, rand_n_e=rand_n_e, rand_V_d=rand_V_d)
else:
    np.savez(f"{save_dir}/{style_str}{extra_str}_snap_{snapshot:d}.npz", rbins=rbins, rand_rbins_hckpc=rand_rbins_hckpc, P_e=P_e, T_e=T_e, n_e=n_e, N_v=N_v, V_d=V_d, rand_P_e=rand_P_e, rand_n_e=rand_n_e, rand_V_d=rand_V_d, inds_halo=inds_halo)

for myrank in range(n_ranks):
    print("don't delete anything for now")
    #os.unlink(f"{save_dir}/{style_str}_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npz")
