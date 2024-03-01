import sys
import os

import numpy as np

"""
python3 combine_y_sph_cyl.py 8 264 yz_zx

python3 combine_y_sph_cyl.py 16 264
python3 combine_y_sph_cyl.py 16 237
python3 combine_y_sph_cyl.py 16 214
python3 combine_y_sph_cyl.py 16 179
"""

n_ranks = int(sys.argv[1]) # 32
snapshot = int(sys.argv[2]) # 179, 214, 237, 264

# select orientation
if len(sys.argv) > 3:
    orientation = sys.argv[3]
    orient_str = f"_{orientation}"
    orientations = orientation.split('_')
else:
    orientation = 'xy'
    orient_str = ""
    orientations = ['xy']

# location stored
save_dir = "/freya/ptmp/mpa/boryanah/data_sz/"
style_str = f"SZ{orient_str}_sph_cyl_r200c_m200c_1e12Msun"

# read the data
data = np.load(f"{save_dir}/{style_str}_snap_{snapshot:d}_rank0_{n_ranks:d}.npz")
inds_halo = data['inds_halo']
N_gal = len(inds_halo)

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

for myrank in range(n_ranks):
    print("myrank", myrank)
    data = np.load(f"{save_dir}/{style_str}_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npz")

    Y_200c_sph += data['Y_200c_sph']
    Y_200c_cyl_xy += data['Y_200c_cyl_xy']
    Y_200c_cyl_yz += data['Y_200c_cyl_yz']
    Y_200c_cyl_zx += data['Y_200c_cyl_zx']
    b_200c_sph += data['b_200c_sph']
    b_200c_cyl_xy += data['b_200c_cyl_xy']
    b_200c_cyl_yz += data['b_200c_cyl_yz']
    b_200c_cyl_zx += data['b_200c_cyl_zx']
    tau_200c_sph += data['tau_200c_sph']
    tau_200c_cyl_xy += data['tau_200c_cyl_xy']
    tau_200c_cyl_yz += data['tau_200c_cyl_yz']
    tau_200c_cyl_zx += data['tau_200c_cyl_zx']

print("trying to save")
np.savez(f"{save_dir}/{style_str}_snap_{snapshot:d}.npz", Y_200c_sph=Y_200c_sph, Y_200c_cyl_xy=Y_200c_cyl_xy, Y_200c_cyl_yz=Y_200c_cyl_yz, Y_200c_cyl_zx=Y_200c_cyl_zx, b_200c_sph=b_200c_sph, b_200c_cyl_xy=b_200c_cyl_xy, b_200c_cyl_yz=b_200c_cyl_yz, b_200c_cyl_zx=b_200c_cyl_zx, tau_200c_sph=tau_200c_sph, tau_200c_cyl_xy=tau_200c_cyl_xy, tau_200c_cyl_yz=tau_200c_cyl_yz, tau_200c_cyl_zx=tau_200c_cyl_zx, inds_halo=inds_halo)

for myrank in range(n_ranks):
    print("don't delete anything for now")
    #os.unlink(f"{save_dir}/{style_str}_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npz")
