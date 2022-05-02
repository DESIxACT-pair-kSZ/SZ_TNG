import numpy as np
import matplotlib.pyplot as plt

dirs = ['xy', 'yz', 'zx']

for i in range(len(dirs)):
    
    Y_xy = np.load(f"data_maps_sz/Y_compton_{dirs[i]:s}.npy")
    std_Y_xy = Y_xy.std()
    mean_Y_xy = Y_xy.mean() 
    print("max = ", Y_xy.max())
    print("min = ", Y_xy.min())
    print("std = ", std_Y_xy)
    print("mean = ", mean_Y_xy)
    
    plt.figure(figsize=(16,14))
    plt.imshow(np.log10(1+Y_xy), vmin=np.log10(1+mean_Y_xy-4.*std_Y_xy), vmax=np.log10(1+mean_Y_xy+4.*std_Y_xy))
    plt.savefig(f"figs/Y_{dirs[i]:s}.png")

    Y_xy = np.load(f"data_maps_sz/b_{dirs[i]:s}.npy")
    std_Y_xy = Y_xy.std()
    mean_Y_xy = Y_xy.mean() 
    print("max = ", Y_xy.max())
    print("min = ", Y_xy.min())
    print("std = ", std_Y_xy)
    print("mean = ", mean_Y_xy)
    
    plt.figure(figsize=(16,14))
    plt.imshow((1+Y_xy), vmin=(1+mean_Y_xy-4.*std_Y_xy), vmax=(1+mean_Y_xy+4.*std_Y_xy))
    plt.savefig(f"figs/b_{dirs[i]:s}.png")
