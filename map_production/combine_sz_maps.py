import sys
import os

import numpy as np

"""
# MTNG
python3 combine_sz_maps.py 1 125

# MTNG
python3 combine_sz_maps.py 32 264
python3 combine_sz_maps.py 32 237
python3 combine_sz_maps.py 32 214
python3 combine_sz_maps.py 32 179

# TNG300
python3 combine_sz_maps.py 30 78
python3 combine_sz_maps.py 30 69
python3 combine_sz_maps.py 30 62
python3 combine_sz_maps.py 30 56
python3 combine_sz_maps.py 30 52

# TNG100
python3 combine_sz_maps.py 32 69
#python3 combine_sz_maps.py 30 62
#python3 combine_sz_maps.py 30 56
#python3 combine_sz_maps.py 30 52


# Illustris
python3 combine_sz_maps.py 32 105
python3 combine_sz_maps.py 32 98
python3 combine_sz_maps.py 32 92
python3 combine_sz_maps.py 32 88

python3 combine_sz_maps.py 32 135
python3 combine_sz_maps.py 32 103


python3 combine_sz_maps.py 32 114 Illustris
python3 combine_sz_maps.py 30 78 TNG300

"""

n_ranks = int(sys.argv[1]) # 32
snapshot = int(sys.argv[2]) # 179, 214, 237, 264
sim_name = sys.argv[3]
want_all_dir = False
want_3d = False

# location stored
if sim_name == "MTNG":
    save_dir = "/freya/ptmp/mpa/boryanah/data_sz/" # MTNG
elif sim_name == "SIMBA":
    save_dir = "/freya/ptmp/mpa/boryanah/SIMBA100/" # SIMBA
elif sim_name == "TNG300":
    save_dir = "/freya/ptmp/mpa/boryanah/data_sz/TNG300/" # TNG300
elif sim_name == "Illustris":
    save_dir = "/freya/ptmp/mpa/boryanah/data_sz/Illustris/" # Illustris
elif sim_name == "TNG100":
    save_dir = "/freya/ptmp/mpa/boryanah/data_sz/TNG100/" # TNG100

if want_3d:
    fields = ["Y_compton_3d", "b_xy_3d", "tau_3d"]
else:    
    if want_all_dir:
        #fields = ["Y_compton_xy", "b_xy", "tau_xy", "Y_compton_yz", "b_yz", "tau_yz", "Y_compton_zx", "b_zx", "tau_zx"]
        fields = ["Y_compton_yz", "b_yz", "tau_yz", "Y_compton_zx", "b_zx", "tau_zx"] 
    else:
        fields = ["Y_compton_xy", "b_xy", "tau_xy"]

# todo could put somewhere nicer
if "Illustris" in save_dir or "TNG100" in save_dir or "SIMBA" in save_dir:
    nbins = 2001
else:
    nbins = 10001
if "Illustris" in save_dir:
    Ndim = 910
else:
    Ndim = 512#1080 #1024
    
for i in range(len(fields)):
    field = fields[i]
    print(field)
    for myrank in range(n_ranks):
        print("myrank", myrank)
        if myrank == 0:
            if want_3d:
                field_arr = np.zeros((Ndim, Ndim, Ndim), dtype=np.float32)
            else:
                field_arr = np.zeros((nbins-1, nbins-1), dtype=np.float32)
        field_arr += np.load(f"{save_dir}/{field}_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npy")

    print("trying to save")
    np.save(f"{save_dir}/{field}_snap_{snapshot:d}.npy", field_arr)

for i in range(len(fields)):
    field = fields[i]
    print(field)
    for myrank in range(n_ranks):
        #print("don't delete anything for now")
        os.unlink(f"{save_dir}/{field}_snap_{snapshot:d}_rank{myrank:d}_{n_ranks:d}.npy")
