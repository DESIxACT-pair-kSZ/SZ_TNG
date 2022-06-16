"""
Script for obtaining SZ maps in three projections
"""
import time
import numpy as np
import h5py
from tools import numba_tsc_3D, hist2d_numba_seq
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import argparse
import math

# command line arguments -- latin hypercube number, initial and final snapshot number, number of bins for maps 
# note: final snapshot number is the highest redshift for lightcone

parser = argparse.ArgumentParser()
parser.add_argument('--LHnum','-lh', type=str)
parser.add_argument('--snapinit','-si', type=int)
parser.add_argument('--snapfinal','-sf', type=int)
parser.add_argument('--nbins','-nb', type=int)
args = parser.parse_args()
LHnum = args.LHnum
snapinit = args.snapinit
snapfinal = args.snapfinal
nbins = args.nbins

# check to see that LH and snap nums are valid
LHlist = [str(i) for i in range(1000)]
snaplist = np.arange(34)

if LHnum not in LHlist:
    print('LH number out of bounds')
save_dir = "/global/cscratch1/sd/kjc268/CAMELS_TSZ/LH_{}".format(LHnum) 

if snapinit not in snaplist: 
    print('snapshot number out of bounds')
if snapfinal in snaplist: 
    if snapfinal > snapinit: 
        print('final snapshot number must be less than initial snapshot number')
    elif snapfinal == snapinit:
        print('no lightcone, single redshift maps')
else: 
    snapfinal = snapinit 
    print('no lightcone, single redshift maps')
    
# physics constants in cgs
k_B = 1.3807e-16 # cgs (erg/K)
sigma_T = 6.6524587158e-29*1.e2**2 # cm^2
m_e = 9.10938356e-28 # g
c = 29979245800. # cm/s
const = k_B*sigma_T/(m_e*c**2) # cgs (cm^2/K)
kpc_to_cm = ((1.*u.kpc).to(u.cm)).value # cm /kpc

snaps = np.arange(snapfinal,snapinit+1)
af = 0 # initialize smallest scale factor
Lbox = 25. # cMpc/h
Lbox *= 1000. # ckpc/h -- kpc for binning
# for the histograming
nbins2d = (nbins-1, nbins-1)
nbins2d = np.asarray(nbins2d).astype(np.int64)
# bins
bins = np.linspace(0, Lbox, nbins)
binc = (bins[1:]+bins[:-1])*0.5
# initialize empty maps
y_xy = np.zeros((nbins-1, nbins-1))
y_yz = np.zeros((nbins-1, nbins-1))
y_zx = np.zeros((nbins-1, nbins-1))
b_xy = np.zeros((nbins-1, nbins-1))
b_yz = np.zeros((nbins-1, nbins-1))
b_zx = np.zeros((nbins-1, nbins-1))

# loop over each snapshot
for snapshot in snaps: 
    print("snapshot =", snapshot)
    basePath = "/global/cscratch1/sd/kjc268/CAMELS_ITNG/LH_{}/snap_{}.hdf5".format(LHnum,str(snapshot).zfill(3))
    n_chunks = h5py.File(basePath)['Header'].attrs['NumFilesPerSnapshot']
    z = h5py.File(basePath)['Header'].attrs['Redshift'] # TNG
    a = 1./(1+z)
    if snapshot == snapfinal: 
        af = a

    # sim params (same for MTNG and TNG)
    h = h5py.File(basePath)['Header'].attrs['HubbleParam']
    solar_mass = 1.989e33 # g/msun
    unit_mass = solar_mass*1e10/h # h g/ 10^10 msun
    unit_vol = (kpc_to_cm/h)**3 # (cm/kpc/h)^3
    unit_dens = unit_mass/unit_vol # h g/(10^10 msun) / (cm/kpc/h)^3 

    # loop over each chunk
    for i in range(0, n_chunks):
        print("chunk = ", i, end='\r')

        # read saved fields of interest
        Te = np.load(f"{save_dir}/temperature_chunk_{i:d}_snap_{snapshot:d}.npy") # K
        ne = np.load(f"{save_dir}/number_density_chunk_{i:d}_snap_{snapshot:d}.npy") # ccm^-3
        Ve = np.load(f"{save_dir}/velocity_chunk_{i:d}_snap_{snapshot:d}.npy")*1.e5 # cm/s from km/s
        dV = np.load(f"{save_dir}/volume_chunk_{i:d}_snap_{snapshot:d}.npy") # (ckpc/h)^3 (TNG)
        C = np.load(f"{save_dir}/position_chunk_{i:d}_snap_{snapshot:d}.npy") # ckpc/h (TNG)

        # compute the contribution to the y and b signals of each cell
        # unit_vol converts ckpc/h^3 to ccm^3, ne is in comoving units so ne*dV*unit_vol completely cancel
        # divide by proper box area in centimeters (aLbox is kpc/h so (aLbox*unit_length)^-2 is cm^-2 and units cancel with sigma_T
        ne = np.array(ne, dtype=np.float64)
        Te = np.array(Te, dtype=np.float64)
        dV = np.array(dV, dtype=np.float64)
        
        dY = const*(ne*Te*dV)*unit_vol/(a*Lbox*unit_vol**(1/3))**2
        b = sigma_T*(ne[:, None]*(Ve/c)*dV[:, None])*unit_vol/(a*Lbox*unit_vol**(1/3))**2
        
        # lightcone if multiple snapshots 
        d = int(np.round((af/a)*Lbox/2)) # ckpc/h
        center = math.floor(Lbox/2)
        start = center-d
        end = center+d
        mx = np.ma.masked_inside(C[:,0], start, end) 
        my = np.ma.masked_inside(C[:,1], start, end)
        mz = np.ma.masked_inside(C[:,2], start, end)
        # points that fall outside of af box
        mxy = mx.mask*my.mask
        myz = my.mask*mz.mask
        mzx = mz.mask*mx.mask
        
        ranges = ((start, end),(start, end)) #ckpc/h
        ranges = np.asarray(ranges).astype(np.float64)
        # flatten out in each of three directions (10 times faster than histogramdd
        Dxy = hist2d_numba_seq(np.array([C[:, 0][mxy], C[:, 1][mxy]]), bins=nbins2d, ranges=ranges, weights=dY[mxy], dtype=np.float64)
        Dyz = hist2d_numba_seq(np.array([C[:, 1][myz], C[:, 2][myz]]), bins=nbins2d, ranges=ranges, weights=dY[myz], dtype=np.float64)
        Dzx = hist2d_numba_seq(np.array([C[:, 2][mzx], C[:, 0][mzx]]), bins=nbins2d, ranges=ranges, weights=dY[mzx], dtype=np.float64)

        # coadd the contribution from this chunk to the three projections
        y_xy += Dxy
        y_yz += Dyz
        y_zx += Dzx

        # flatten out in each of three directions
        Dxy = hist2d_numba_seq(np.array([C[:, 0][mxy], C[:, 1][mxy]]), bins=nbins2d, ranges=ranges, weights=b[:, 2][mxy], dtype=np.float64)
        Dyz = hist2d_numba_seq(np.array([C[:, 1][myz], C[:, 2][myz]]), bins=nbins2d, ranges=ranges, weights=b[:, 0][myz], dtype=np.float64)
        Dzx = hist2d_numba_seq(np.array([C[:, 2][mzx], C[:, 0][mzx]]), bins=nbins2d, ranges=ranges, weights=b[:, 1][mzx], dtype=np.float64)

        # coadd the contribution from this chunk to the three projections
        b_xy += Dxy
        b_yz += Dyz
        b_zx += Dzx
                
if snapfinal == snapinit: 
    snapinit = str(snapinit).zfill(3)
else: 
    snapinit = str(snapinit).zfill(3) + "_to_" + str(snapfinal).zfill(3)

# save tSZ maps
np.save(f"{save_dir}/Y_compton_xy_snap_{snapinit:s}.npy", y_xy)
np.save(f"{save_dir}/Y_compton_yz_snap_{snapinit:s}.npy", y_yz)
np.save(f"{save_dir}/Y_compton_zx_snap_{snapinit:s}.npy", y_zx)

# save kSZ maps
np.save(f"{save_dir}/b_xy_snap_{snapinit:s}.npy", b_xy)
np.save(f"{save_dir}/b_yz_snap_{snapinit:s}.npy", b_yz)
np.save(f"{save_dir}/b_zx_snap_{snapinit:s}.npy", b_zx)
