import sys
import os

import numpy as np

"""
python3 combine_dm_profile_combo.py 32 264 fp
python3 combine_dm_profile_combo.py 32 237 fp
python3 combine_dm_profile_combo.py 32 214 fp
python3 combine_dm_profile_combo.py 32 179 fp

python3 combine_dm_profile_combo.py 32 264 dm
python3 combine_dm_profile_combo.py 32 237 dm
python3 combine_dm_profile_combo.py 32 214 dm
python3 combine_dm_profile_combo.py 32 179 dm
"""

n_ranks = int(sys.argv[1]) # 32
snapshot = int(sys.argv[2]) # 179, 214, 237, 264
type_sim = f"_{sys.argv[3]}" # "_fp", "_dm"

# location stored
save_dir = "/freya/ptmp/mpa/boryanah/data_sz/"
style_str = f"prof_dm{type_sim}_sph_r200c_m200c_1e12Msun"

# read the data
data = np.load(f"{save_dir}/{style_str}_snap_{snapshot:d}_rank0_{n_ranks:d}.npz")
inds_halo = data['inds_halo']
N_gal = len(inds_halo)
rbins = data['rbins']
print("read the first one")

# initialize
rho_dm = np.zeros((N_gal, len(rbins)-1))
V_d = np.zeros((N_gal, len(rbins)-1))
N_v = np.zeros((N_gal, len(rbins)-1)) 

for myrank in range(n_ranks):
    print("myrank", myrank)
    data = np.load(f"{save_dir}/{style_str}_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npz")
    rho_dm += data['rho_dm']
    V_d += data['V_d']
    N_v += data['N_v']

print("trying to save")
np.savez(f"{save_dir}/{style_str}_snap_{snapshot:d}.npz", rbins=rbins, rho_dm=rho_dm, N_v=N_v, V_d=V_d, inds_halo=inds_halo)

for myrank in range(n_ranks):
    print("don't delete anything for now")
    #os.unlink(f"{save_dir}/{style_str}_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npz")
