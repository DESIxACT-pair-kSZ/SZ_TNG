# is Te same as T gas cells
import numpy as np
import h5py
from tools import numba_tsc_3D
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# physics constants in cgs
gamma = 5/3.
k_B = 1.3807e-16 # cgs (erg/K)
m_p = 1.6726e-24 # g
unit_c = 1.023**2*1.e10 # see faq kpc/Gyr to km/s (Unit_Length/Unit_Time)^2*Unit_Mass Mpc?!?
X_H = 0.76
sigma_T = 6.6524587158e-29*1.e2**2 # cm^2
m_e = 9.10938356e-28 # g
c = 29979245800. # cm/s
const = k_B*sigma_T/(m_e*c**2)
kpc_to_cm = ((1.*u.kpc).to(u.cm)).value # cm
solar_mass = 1.989e33 # g

# sim params  these are the latest? according to params-unused
h = 67.74/100.
unit_mass = 1.e10*(solar_mass/h)
unit_dens = 1.e10*(solar_mass/h)/(kpc_to_cm/h)**3 # g/cm**3 # note density has units of h in it
unit_vol = (kpc_to_cm/h)**3


# path to simulation
basePath = "/virgo/simulations/MTNG/Hydro-Arepo/MTNG-L500-4320-A/"; type_sim = "_fp"; snapshot = 179; n_chunks = 640; z = 1.
#basePath = "/virgo/simulations/MTNG/DM-Gadget4/MTNG-L500-4320-A/"; type_sim = "_dm"; snapshot = 184; n_chunks = 128; z = 1.

# sim info
n_total = 4320**3
Lbox = 500. # Mpc/h
PartType = 'PartType0'
a = 1./(1+z)

"""
# angular distance
cosmo = FlatLambdaCDM(H0=h*100., Om0=0.3089, Tcmb0=2.725)
d_L = cosmo.luminosity_distance(redshift).to(u.kpc).value
d_A = d_L/(1.+redshift)**2 # dA = dL/(1+z)^2 # kpc
d_A *= kpc_to_cm # cm
"""

# fields to load
fields = ['InternalEnergy', 'ElectronAbundance', 'Density', 'Masses', 'Coordinates', 'Velocities']
#Tmin = 1.e6
Tmin = 0. #1.e4
#Tmin = 5.e6 # attention initial files used this

for i in range(0, n_chunks):
    print("chunk = ", i, end='\r')

    # read positions of DM particles
    hfile = h5py.File(basePath+f'snapdir_{snapshot:03d}/snapshot_{snapshot:03d}.{i:d}.hdf5')[PartType]
    #print(list(hfile.keys()))

    C = hfile['Coordinates'][:]
    EA = hfile['ElectronAbundance'][:]
    IE = hfile['InternalEnergy'][:]
    D = hfile['Density'][:]
    M = hfile['Masses'][:]
    C = hfile['Coordinates'][:]
    V = hfile['Velocities'][:]
    
    dV = M/D # cMpc/h^3
    D *= unit_dens # g/cm^3

    Te = (gamma - 1.)*IE/k_B * 4*m_p/(1 + 3*X_H + 4*X_H*EA) * unit_c # K
    ne = EA*X_H*D/m_p # cm^-3
    Ve = V*np.sqrt(a) # km/s

    choice = Te > Tmin
    print("percentage interesting = ", np.sum(choice)*100./len(choice))
    Te = Te[choice]
    ne = ne[choice]
    Ve = Ve[choice]
    dV = dV[choice]
    C = C[choice]


    #print("mean temperature = ", np.mean(Te))
    #if np.sum(Te > 1.e6) == 0: break
    
    np.save(f"/freya/ptmp/mpa/boryanah/data_sz/temperature_chunk_{i:d}_snap_{snapshot:d}.npy", Te)
    np.save(f"/freya/ptmp/mpa/boryanah/data_sz/number_density_chunk_{i:d}_snap_{snapshot:d}.npy", ne)
    np.save(f"/freya/ptmp/mpa/boryanah/data_sz/velocity_chunk_{i:d}_snap_{snapshot:d}.npy", Ve)
    np.save(f"/freya/ptmp/mpa/boryanah/data_sz/volume_chunk_{i:d}_snap_{snapshot:d}.npy", dV)
    np.save(f"/freya/ptmp/mpa/boryanah/data_sz/position_chunk_{i:d}_snap_{snapshot:d}.npy", C)
