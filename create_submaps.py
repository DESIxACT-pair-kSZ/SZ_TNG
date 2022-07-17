# are the std's used?
# two things that don't make sense: if i == 0; and then for i in len(ras)
# tau and y
# periodic bc
# div matter?
# disk and not ring: leads to big difference (a couple of times)
# beam ор no beam?
import numpy as np
import matplotlib.pyplot as plt
from pixell import enmap, utils, enplot
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from tools import extractStamp, calc_T_AP

k_B = 1.380649e-23 # J/K, Boltzmann constant
h_P =  6.62607015e-34 # Js, Planck's constant
T_CMB = 2.7255 # K

def f_nu(nu):
    """ f(ν) = x coth (x/2) − 4, with x = hν/kBTCMB """
    x = h_P*nu/(k_B*T_CMB)
    f = x / np.tanh(x/2.) - 4.
    return f

def eshow(x, fn, **kwargs):
    ''' Define a function to help us plot the maps neatly '''
    plots = enplot.get_plots(x, **kwargs)
    #enplot.show(plots, method = "python")
    enplot.write("figs/"+fn, plots)

def compute_aperture(mp, msk, ra, dec, Th_arcmin, r_max_arcmin, resCutoutArcmin, projCutout):
    
    # extract stamp
    _, stampMap, stampMask = extractStamp(mp, ra, dec, rApMaxArcmin=r_max_arcmin, resCutoutArcmin=resCutoutArcmin, projCutout=projCutout, pathTestFig='figs/', test=False, cmbMask=msk)
    
    # skip if mask is zero everywhere
    if (np.sum(stampMask) == 0.) or (stampMask is None): return 0., 0., 0., 0.

    # record T_AP
    dT_i, dT_o, dT_i_std, dT_o_std = calc_T_AP(stampMap, Th_arcmin, mask=stampMask, divmap=None)

    return dT_i, dT_o, dT_i_std, dT_o_std

# parameters
nu_ACT = 150.e9 # Hz, f150 (150 GHz) and f090 (98 GHz)
save_dir = "/mnt/marvin1/boryanah/SZ_TNG/"
field_dir = "/mnt/gosling1/boryanah/TNG300/"
beam_fwhm = 2.1 # arcmin
theta_arcmin = 2.1 # arcmin
orientation = "xy"
snapshot = 67
z = 0.5
N_gal = 10000
Lbox = 205000. # ckpc/h
projCutout = 'cea'
resCutoutArcmin = 0.05 # resolution
rApMaxArcmin = theta_arcmin

# define cosmology
h = 0.6774
cosmo = FlatLambdaCDM(H0=h*100., Om0=0.3089, Tcmb0=2.725)

# compute angular distance
d_L = cosmo.luminosity_distance(z).to(u.Mpc).value
d_A = d_L/(1.+z)**2 # dA = dL/(1+z)^2 # Mpc
d_A *= h # Mpc/h                             

# get size on the sky of each pixel at given redshift
a = 1./(1+z)
Lbox_deg = np.sqrt((a*Lbox/1000.)**2/d_A**2*(180./np.pi)**2)
print("Lbox deg = ", Lbox_deg)

# load y compton, tau and b maps
Y = np.load(save_dir+f"Y_compton_{orientation}_snap_{snapshot:d}.npy")*1.e8 # bc of mistake
b = np.load(save_dir+f"b_{orientation}_snap_{snapshot:d}.npy")*1.e8 # bc of mistake
tau = np.load(save_dir+f"tau_{orientation}_snap_{snapshot:d}.npy")*1.e8 # bc of mistake
f_ACT = f_nu(nu_ACT)

# cell size
N_cell = Y.shape[0]
cell_size = Lbox/N_cell

# box size in degrees   
cell_size_deg = Lbox_deg/N_cell

# create pixell map
box = np.array([[0., 0.],[Lbox_deg, Lbox_deg]]) * utils.degree
shape, wcs = enmap.geometry(pos=box, res=cell_size_deg * utils.degree, proj='car')
tau_map = enmap.zeros(shape, wcs=wcs)
y_map = tau_map.copy()
b_map = tau_map.copy()
tau_map[:] = tau
y_map[:] = Y
b_map[:] = b

# load subhalo fields
SubhaloPos = np.load(field_dir+f"SubhaloPos_{snapshot:d}_fp.npy") # ckpc/h
SubhaloMst = np.load(field_dir+f"SubhaloMassType_{snapshot:d}_fp.npy")[:, 4]*1.e10 # Msun/h

# select most stellar massive subhalos
i_sort = (np.argsort(SubhaloMst)[::-1])[:N_gal]
pos = SubhaloPos[i_sort]
mstar = SubhaloMst[i_sort]
pos *= cell_size_deg/cell_size
DEC = pos[:, 0]
RA = pos[:, 1]

# compute the aperture photometry for each galaxy
want_plot = False
if want_plot:
    r = 0.5 * utils.arcmin
    srcs = ([DEC*utils.degree, RA*utils.degree])
    mask = enmap.distance_from(shape, wcs, srcs, rmax=r) >= r
    eshow(tau_map * mask, 'galaxies', **{"colorbar":True, "ticks": 5, "downgrade": 4})
    plt.close()

# mask which in this case is just ones
msk = tau_map.copy()
msk *= 0.
msk += 1.

# for each galaxy, create a submap and select mean and so on
tau_inns = np.zeros(len(RA))
tau_outs = np.zeros(len(RA))
y_inns = np.zeros(len(RA))
y_outs = np.zeros(len(RA))
#tau_inn_stds = np.zeros(len(RA))
#tau_out_stds = np.zeros(len(RA))
for i in range(len(RA)):
    tau_inn, tau_out, tau_inn_std, tau_out_std = compute_aperture(tau_map, msk, RA[i], DEC[i], theta_arcmin, rApMaxArcmin, resCutoutArcmin, projCutout)
    tau_inns[i] = tau_inn
    tau_outs[i] = tau_out
    y_inn, y_out, y_inn_std, y_out_std = compute_aperture(y_map, msk, RA[i], DEC[i], theta_arcmin, rApMaxArcmin, resCutoutArcmin, projCutout)
    y_inns[i] = y_inn
    y_outs[i] = y_out
    #tau_inn_stds[i] = tau_inn_std
    #tau_out_stds[i] = tau_out_std
    if i % 100 == 0: print("i, ra, dec, inner, outer, std = ", i, RA[i], DEC[i], tau_inn, tau_out, y_inns, y_outs)
np.savez(f"data/galaxies_top{N_gal:d}_{orientation}_{snapshot:d}_fp.npz", RA=RA, DEC=DEC, mstar=mstar, tau_disks=tau_inns, tau_rings=tau_outs, y_disks=y_inns, y_rings=y_outs)#, disk_stds=tau_inn_stds, ring_stds=tau_out_stds)

"""
# find comoving distance at that redshift
d_c = # ckpc/h
pos[:, 2] += d_c
dist = np.sqrt(np.sum(pos**2., axis=1))
assert pos.shape[0] == len(dist)
unit_pos = pos / dist
"""

    
