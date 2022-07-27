# Eve's code
# two things that don't make sense: if i == 0; and then for i in len(ras)
# tau and y

# improvements:
# are the std's used? (could be used for fitting)
# periodic bc (could improve things a bit)
# how to quantify error on prediction?

# realism:
# beam ор no beam? (could simulate this for realism)
# div matter? (could simulate this for realism)
# disk and not ring: leads to big difference (a couple of times)

# future:
# so goal of this would be to redo Nick's analysis for different apertures and different samples and go to lower masses
# SBPL for clusters
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pixell import enmap, utils, enplot
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.coordinates import SkyCoord

from tools import extractStamp, calc_T_AP#, get_tzav_fast

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
aperture_mode = "r500"
#aperture_mode = "fixed"
if aperture_mode == "fixed":
    theta_arcmin = 1.3 #2.1 # arcmin
elif aperture_mode == "r500":
    theta_arcmin = None
orientation = "xy"
snapshot = 67; z = 0.5
#snapshot = 78; z = 0.3
#snapshot = 91; z = 0.1
Lbox = 205000. # ckpc/h
n_gal = 300/165000**3. # Nick Battaglia density in (ckpc/h)^-3
#galaxy_choice = "star_mass"
galaxy_choice = "halo_mass"
#N_gal = int(np.round(n_gal*Lbox**3))
N_gal = 30000 # mean
#N_gal = 31000 # integrated
projCutout = 'cea'
resCutoutArcmin = 0.05 # resolution
rApMaxArcmin = theta_arcmin

# define cosmology
h = 0.6774
cosmo = FlatLambdaCDM(H0=h*100., Om0=0.3089, Tcmb0=2.725)

# compute angular distance
d_L = cosmo.luminosity_distance(z).to(u.Mpc).value
d_C = d_L/(1.+z)
d_A = d_L/(1.+z)**2 # dA = dL/(1+z)^2 # Mpc
d_A *= h # Mpc/h
d_C *= h # Mpc/h
print("comoving distance = ", d_C)

# get size on the sky of each pixel at given redshift
a = 1./(1+z)
Lbox_deg = np.sqrt((a*Lbox/1000.)**2/d_A**2*(180./np.pi)**2) # degrees
print("Lbox deg = ", Lbox_deg)

# load y compton, tau and b maps
Y = np.load(save_dir+f"Y_compton_{orientation}_snap_{snapshot:d}.npy")*1.e8 # bc of mistake
b = np.load(save_dir+f"b_{orientation}_snap_{snapshot:d}.npy")*1.e8 # bc of mistake
tau = np.load(save_dir+f"tau_{orientation}_snap_{snapshot:d}.npy")*1.e8 # bc of mistake
f_ACT = f_nu(nu_ACT)

# cell size
N_cell = Y.shape[0]
cell_size = Lbox/N_cell # ckpc/h

# box size in degrees   
cell_size_deg = Lbox_deg/N_cell

# create pixell map
box = np.array([[0., 0.],[Lbox_deg, Lbox_deg]]) * utils.degree
shape, wcs = enmap.geometry(pos=box, res=cell_size_deg * utils.degree, proj='car')
tau_map = enmap.zeros(shape, wcs=wcs)
y_map = enmap.zeros(shape, wcs=wcs)
b_map = enmap.zeros(shape, wcs=wcs)
tau_map[:] = tau
y_map[:] = Y
b_map[:] = b

# load subhalo fields
if galaxy_choice == "star_mass":
    SubhaloPos = np.load(field_dir+f"SubhaloPos_{snapshot:d}_fp.npy") # ckpc/h
    SubhaloVel = np.load(field_dir+f"SubhaloVel_{snapshot:d}_fp.npy") # km/s
    SubhaloMst = np.load(field_dir+f"SubhaloMassType_{snapshot:d}_fp.npy")[:, 4]*1.e10 # Msun/h

    # select most stellar massive subhalos
    i_sort = (np.argsort(SubhaloMst)[::-1])[:N_gal]
    pos = SubhaloPos[i_sort]
    vel = SubhaloVel[i_sort]
    mstar = SubhaloMst[i_sort]
    
elif galaxy_choice == "halo_mass":
    GroupPos = np.load(field_dir+f"GroupPos_{snapshot:d}_fp.npy") # ckpc/h
    GroupVel = np.load(field_dir+f"GroupVel_{snapshot:d}_fp.npy")/a # km/s
    Group_M_Crit500 = np.load(field_dir+f"Group_M_Crit500_{snapshot:d}_fp.npy")*1.e10 # Msun/h
    Group_R_Crit500 = np.load(field_dir+f"Group_R_Crit500_{snapshot:d}_fp.npy")*a # kpc/h
    # select most stellar massive subhalos
    i_sort = (np.argsort(Group_M_Crit500)[::-1])[:N_gal]
    pos = GroupPos[i_sort]
    vel = GroupVel[i_sort]
    r500 = Group_R_Crit500[i_sort]
    if aperture_mode == "r500":
        theta_arcmin = r500/(1000.*h)/d_A*180.*60/np.pi
        #theta_arcmin = np.float(f"{theta_arcmin:.1f}")
        print("min, max, mean, median theta arcmin = ", np.min(theta_arcmin), np.max(theta_arcmin), np.mean(theta_arcmin), np.median(theta_arcmin))
        rApMaxArcmin = theta_arcmin.max()
    mstar = Group_M_Crit500[i_sort] # not mstar!
    print(f"lowest halo mass = {np.min(mstar):.2e}")

np.save("vel.npy", vel) # TESTING!!!!!!!!!!!!
quit()
# convert comoving distance to redshift
pos_deg = (pos*cell_size_deg/cell_size) # degrees
DEC = pos_deg[:, 0]
RA = pos_deg[:, 1]
print("RA", RA.min(), RA.max())
print("DEC", DEC.min(), DEC.max())

"""
hist_dec, bins_dec = np.histogram(DEC, bins=101)
hist_ra, bins_ra = np.histogram(RA, bins=101)
binc_dec = (bins_dec[1:] + bins_dec[:-1])*.5
binc_ra = (bins_ra[1:] + bins_ra[:-1])*.5
plt.plot(binc_dec, hist_dec, ls='--', label='DEC')
plt.plot(binc_ra, hist_ra, ls='--', label='RA')

def cart_to_eq(pos, frame='galactic'):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    sc = SkyCoord(x, y, z, representation_type='cartesian', frame=frame)
    scg = sc.transform_to(frame='icrs')
    scg.representation_type = 'unitspherical'

    ra, dec = scg.ra.value, scg.dec.value
    return ra, dec

# before this pos is comoving (kpc/h)
pos[:, 2] += d_C*1000. - Lbox/2.
chi = np.linalg.norm(pos, axis=1)
unit_pos = pos/chi[:, None]
theta, phi = hp.vec2ang(unit_pos)
RA = phi*180./np.pi
DEC = (np.pi/2. - theta)*180./np.pi
print("RA", RA.min(), RA.max())
print("DEC", DEC.min(), DEC.max())

hist_dec, bins_dec = np.histogram(DEC, bins=101)
hist_ra, bins_ra = np.histogram(RA, bins=101)
binc_dec = (bins_dec[1:] + bins_dec[:-1])*.5
binc_ra = (bins_ra[1:] + bins_ra[:-1])*.5
plt.plot(binc_dec, hist_dec, ls='-', label='DEC')
plt.plot(binc_ra, hist_ra, ls='-', label='RA')

from nbodykit.transform import CartesianToEquatorial
#RA, DEC = CartesianToEquatorial(pos/1000., frame='icrs')
RA, DEC = cart_to_eq(pos/1000., frame='icrs')
RA, DEC = np.asarray(RA), np.asarray(DEC)
print("RA", RA.min(), RA.max())
print("DEC", DEC.min(), DEC.max())
hist_dec, bins_dec = np.histogram(DEC, bins=101)
hist_ra, bins_ra = np.histogram(RA, bins=101)
binc_dec = (bins_dec[1:] + bins_dec[:-1])*.5
binc_ra = (bins_ra[1:] + bins_ra[:-1])*.5
plt.plot(binc_dec, hist_dec, ls=':', label='DEC')
plt.plot(binc_ra, hist_ra, ls=':', label='RA')
plt.legend()
plt.show()
"""

# compute the aperture photometry for each galaxy
want_plot = False
if want_plot:
    r = 0.7 * utils.arcmin
    srcs = ([DEC*utils.degree, RA*utils.degree])
    mask = enmap.distance_from(shape, wcs, srcs, rmax=r) >= r
    """
    mask = enmap.distance_from(shape, wcs, ([DEC[0:1]*utils.degree, RA[0:1]*utils.degree]), rmax=theta_arcmin[0:1] * utils.arcmin) >= (theta_arcmin[0:1] * utils.arcmin)
    for i in range(1, len(theta_arcmin)):
        print(i)
        mask *= enmap.distance_from(shape, wcs, ([DEC[i:(i+1)]*utils.degree, RA[i:(i+1)]*utils.degree]), rmax=theta_arcmin[i:(i+1)] * utils.arcmin) >= (theta_arcmin[i:(i+1)] * utils.arcmin)
    """
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
b_inns = np.zeros(len(RA))
b_outs = np.zeros(len(RA))
for i in range(len(RA)):
    tau_inn, tau_out, tau_inn_std, tau_out_std = compute_aperture(tau_map, msk, RA[i], DEC[i], theta_arcmin[i], rApMaxArcmin, resCutoutArcmin, projCutout)
    tau_inns[i] = tau_inn
    tau_outs[i] = tau_out
    y_inn, y_out, y_inn_std, y_out_std = compute_aperture(y_map, msk, RA[i], DEC[i], theta_arcmin[i], rApMaxArcmin, resCutoutArcmin, projCutout)
    y_inns[i] = y_inn
    y_outs[i] = y_out
    b_inn, b_out, b_inn_std, b_out_std = compute_aperture(b_map, msk, RA[i], DEC[i], theta_arcmin[i], rApMaxArcmin, resCutoutArcmin, projCutout)
    b_inns[i] = b_inn
    b_outs[i] = b_out
    if i % 100 == 0: print("i, ra, dec, inner, outer, std = ", i, RA[i], DEC[i], tau_inn, tau_out, y_inn, y_out, b_inn, b_out)
if aperture_mode == "r500":
    np.savez(f"data/galaxies_AP{aperture_mode}_top{N_gal:d}_{orientation}_{snapshot:d}_fp.npz", RA=RA, DEC=DEC, mstar=mstar, pos=pos, vel=vel, tau_disks=tau_inns, tau_rings=tau_outs, y_disks=y_inns, y_rings=y_outs, b_disks=b_inns, b_rings=b_outs)
elif aperture_mode == "fixed":
    np.savez(f"data/galaxies_th{theta_arcmin:.1f}_top{N_gal:d}_{orientation}_{snapshot:d}_fp.npz", RA=RA, DEC=DEC, mstar=mstar, pos=pos, vel=vel, tau_disks=tau_inns, tau_rings=tau_outs, y_disks=y_inns, y_rings=y_outs, b_disks=b_inns, b_rings=b_outs)
