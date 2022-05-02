# is Te same as T gas cells
import numpy as np
import illustris_python as il

# save directory
savePath = ''

# path to simulation
basePath = "/virgo/simulations/MTNG/Hydro-Arepo/MTNG-L500-4320-A/"; type_sim = "_fp"; snapshot = 179; n_chunks = 640; z = 1.
#basePath = "/virgo/simulations/MTNG/DM-Gadget4/MTNG-L500-4320-A/"; type_sim = "_dm"; snapshot = 184; n_chunks = 128; z = 1.

# sim info
n_total = 4320**3
ngrid = 1024
Lbox = 500. # Mpc/h
PartType = 'PartType0'
a = 1./(1+z)

# fields to load
fields = ['InternalEnergy', 'ElectronAbundance', 'Density', 'Masses', 'Coordinates', 'Velocities']

halo_inds = np.arange(0, 1)

for i in range(len(halo_inds)):
    halo_index = halo_inds[i]
    gas = il.snapshot.loadHalo(basePath, snapshot, halo_index, 'gas', fields)
    for field in fields:
        np.save(savePath+f"data_cells/gas_{field:s}_hi{halo_index:04d}_snap{snapshot:d}.npy", gas[field])
