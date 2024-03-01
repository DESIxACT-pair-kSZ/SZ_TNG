import os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""
python3 plot_sz_maps.py 179
python3 plot_sz_maps.py 214
python3 plot_sz_maps.py 237
python3 plot_sz_maps.py 264
"""

snapshot = int(sys.argv[1]) # 179, 214, 237, 264
want_all_dir = False

# location stored
save_dir = "/freya/ptmp/mpa/boryanah/data_sz/"

if want_all_dir:
    fields = ["Y_compton_xy", "b_xy", "tau_xy", "Y_compton_yz", "b_yz", "tau_yz", "Y_compton_zx", "b_zx", "tau_zx"]
else:
    fields = ["b_xy"] #["Y_compton_xy", "b_xy", "tau_xy"]

for i in range(len(fields)):
    field = fields[i]
    print(field)

    field_arr = np.load(f"{save_dir}/{field}_snap_{snapshot:d}.npy")
    
    # plot and save unbeamed
    plt.figure(figsize=(16,14))
    mean_field_arr = np.mean(field_arr)
    std_field_arr = np.std(field_arr)

    if field[0] == "b":
        plt.imshow((field_arr), vmin=(mean_field_arr-4.*std_field_arr), vmax=(mean_field_arr+4.*std_field_arr))
    else:
        plt.imshow(np.log10(1+field_arr), vmin=np.log10(1+mean_field_arr-4.*std_field_arr), vmax=np.log10(1+mean_field_arr+4.*std_field_arr))
    
    plt.savefig(f"figs/{field}_snap{snapshot}.png")
    plt.close()
