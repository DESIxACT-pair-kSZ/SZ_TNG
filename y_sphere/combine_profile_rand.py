import sys
import os

import numpy as np

"""
python3 combine_profile_rand.py 8 264
python3 combine_profile_rand.py 8 237
python3 combine_profile_rand.py 8 214
python3 combine_profile_rand.py 8 179
"""

n_ranks = int(sys.argv[1]) # 16, 32
snapshot = int(sys.argv[2]) # 179, 214, 237, 264

# location stored
save_dir = "/freya/ptmp/mpa/boryanah/data_sz/"
style_str = "prof_sph_r200c_m200c_1e12Msun"

# read the data
data = np.load(f"{save_dir}/{style_str}_snap_{snapshot:d}_rank0_{n_ranks:d}.npz")
print("SHOULD ONLY HAVE RANDOMS OR THE REST SHOULD BE ZEROS", data.files)
rand_rbins_hckpc = data['rand_rbins_hckpc']
print("read the first one")

# initialize
N_rand = 5000 # may or may not be ok
rand_V_d = np.zeros((N_rand, len(rand_rbins_hckpc)-1))
rand_P_e = np.zeros((N_rand, len(rand_rbins_hckpc)-1))
rand_n_e = np.zeros((N_rand, len(rand_rbins_hckpc)-1))

for myrank in range(n_ranks):
    print("myrank", myrank)
    data = np.load(f"{save_dir}/{style_str}_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npz")
    rand_V_d += data['rand_V_d']
    rand_P_e += data['rand_P_e']
    rand_n_e += data['rand_n_e']

print("trying to save")
np.savez(f"{save_dir}/{style_str}_rand_snap_{snapshot:d}.npz", rand_rbins_hckpc=rand_rbins_hckpc, rand_P_e=rand_P_e, rand_n_e=rand_n_e, rand_V_d=rand_V_d)

for myrank in range(n_ranks):
    print("don't delete anything for now")
    #os.unlink(f"{save_dir}/{style_str}_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npz")
