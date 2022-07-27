import os
import gc
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import healpy as hp
from pixell import enmap, enplot, utils, reproject
from classy import Class
from astropy.io import fits
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck13
from scipy.interpolate import interp1d
from colossus.lss import bias

from util import eshow


# physics constants in cgs
m_p = 1.6726e-24 # g
X_H = 0.76 # unitless
sigma_T = 6.6524587158e-25 # cm^2
m_e = 9.10938356e-28 # g
c = 29979245800. # cm/s
kpc_to_cm = ((1.*u.kpc).to(u.cm)).value # cm
Mpc_to_cm = kpc_to_cm*1000. # 3.086e+24 cm
solar_mass = 1.989e33 # g
sigma_T_over_m_p = sigma_T / m_p
solar_mass_over_Mpc_to_cm_squared = solar_mass/Mpc_to_cm**2

def get_P_D_A(Cosmo, RA, DEC, Z):
    # transform to cartesian coordinates (checked)
    CX = np.cos(RA*utils.degree)*np.cos(DEC*utils.degree)
    CY = np.sin(RA*utils.degree)*np.cos(DEC*utils.degree)
    CZ = np.sin(DEC*utils.degree)

    # stack together normalized positions
    P = np.vstack((CX, CY, CZ)).T

    # comoving distance to observer and angular size
    CD = np.zeros(len(Z))
    D_A = np.zeros(len(Z))
    for i in range(len(Z)):
        if Z[i] < 0.: continue
        lum_dist = Cosmo.luminosity_distance(Z[i])
        CD[i] = lum_dist/(1.+Z[i]) # Mpc # pretty sure of the units since classylss has Mpc/h and multiplies by h 
        D_A[i] = lum_dist/(1.+Z[i])**2 # Mpc
    P = P*CD[:, None]
    return P, D_A

def load_cmb_sample(cmb_sample, data_dir, source_arcmin, noise_uK, save=False):
    # filename of CMB map
    if cmb_sample == "ACT_BN":
        fn = data_dir+"/cmb_data/tilec_single_tile_BN_cmb_map_v1.2.0_joint.fits" # BN
        msk_fn = data_dir+f"/cmb_data/comb_mask_ACT_BN_compsep_ps_{source_arcmin:.1f}arcmin.fits"
        #msk_fn = data_dir+"/cmb_data/act_dr4.01_s14s15_BN_compsep_mask.fits"
    elif cmb_sample == "ACT_D56":
        fn = data_dir+"/cmb_data/tilec_single_tile_D56_cmb_map_v1.2.0_joint.fits" # D56
        msk_fn = data_dir+f"/cmb_data/comb_mask_ACT_D56_compsep_ps_{source_arcmin:.1f}arcmin.fits"
        #msk_fn = data_dir+"/cmb_data/act_dr4.01_s14s15_D56_compsep_mask.fits"
    elif cmb_sample == "ACT_DR5_f090":
        fn = data_dir+"/cmb_data/act_planck_dr5.01_s08s18_AA_f090_daynight_map.fits" # DR5 f090
        msk_fn = data_dir+f"/cmb_data/comb_mask_ACT_DR5_f090_ivar_{noise_uK:d}uK_ps_{source_arcmin:.1f}arcmin.fits"
    elif cmb_sample == "ACT_DR5_f150":
        fn = data_dir+"/cmb_data/act_planck_dr5.01_s08s18_AA_f150_daynight_map.fits" # DR5 f150
        msk_fn = data_dir+f"/cmb_data/comb_mask_ACT_DR5_f150_ivar_{noise_uK:d}uK_ps_{source_arcmin:.1f}arcmin.fits"
    elif cmb_sample == "Planck":
        fn = data_dir+"/cmb_data/Planck_COM_CMB_IQU-smica_2048_R3.00_uK.fits" # pixell (because "Planck_"), equatorial
        msk_fn = data_dir+"/cmb_data/Planck_HFI_Mask_PointSrc_Gal70.fits" # combined pixell, equatorial
        #fn = data_dir+"/cmb_data/COM_CMB_IQU-smica_2048_R3.00_full.fits" # healpy (because no "Planck_"), galactic
        #msk_fn = data_dir+"/cmb_data/HFI_Mask_PointSrc_Gal70.fits" # combined healpy (equatorial), galactic
        #msk_fn = data_dir+"/cmb_data/HFI_Mask_PointSrc_Gal70_equatorial.fits" # combined healpy, equatorial
        
    """
    # generate Planck map in pixell format
    res_arcmin = 0.5
    DEC_center, RA_center = 0., 0.
    nx = int(180./(res_arcmin/60.)) # DEC
    ny = int(360./(res_arcmin/60.)) # RA
    print("nx, ny = ", nx, ny)
    shape, wcs = enmap.geometry(shape=(nx, ny), res=res_arcmin*utils.arcmin, pos=(DEC_center, RA_center))
    #mp = reproject.enmap_from_healpix(fn, shape, wcs, ncomp=1, unit=1.e-6, lmax=6000, rot="gal,equ")
    msk = hp.read_map(msk_fn)
    msk = reproject.enmap_from_healpix_interp(msk, shape, wcs, interpolate=False, rot=None) # if already in equatorial, rot=None
    #msk = mp.astype(np.float32)
    print("shape, wcs = ", shape, wcs)
    print("box = ", enmap.box(shape, wcs)/utils.degree)
    #enmap.write_fits(data_dir+f"/cmb_data/Planck_COM_CMB_IQU-smica_2048_R3.00_uK.fits", mp)
    enmap.write_fits(data_dir+f"/cmb_data/Planck_HFI_Mask_PointSrc_Gal70.fits", msk)
    quit()
    """
    
    # reading fits file
    mp = enmap.read_fits(fn)

    if "DR5" in cmb_sample:
        mp = mp[0] # three maps are available
        gc.collect()
    elif "Planck" in cmb_sample:
        mp = mp[0] # saved as (1, 10800, 21600)
    if msk_fn is None:
        msk = mp*0.+1.
        msk[mp == 0.] = 0.
    else:
        msk = enmap.read_fits(msk_fn)
        
    if "Planck" in cmb_sample:
        #msk = msk[0] # saved as (1, 10800, 21600)
        msk[msk < 0.5] = 0.
        msk[msk >= 0.5] = 1.
        print(msk.shape)
    print("fsky, msk min, max, mean = ", np.sum(np.isclose(msk, 1.))/np.product(msk.shape), msk.shape, msk.min(), msk.max(), msk.mean())
    
    # save map
    if save:
        fig_name = (fn.split('/')[-1]).split('.fits')[0]
        mp_box = np.rad2deg(mp.box())
        print("decfrom, decto, rafrom, rato = ", mp_box[0, 0], mp_box[1, 0], mp_box[0, 1], mp_box[1, 1])
        eshow(mp, fig_name, **{"colorbar":True, "range": 300, "ticks": 5, "downgrade": 4})
        eshow(msk, fig_name+"_mask", **{"colorbar":True, "ticks": 5, "downgrade": 4})
        plt.close()
    return mp, msk

def load_galaxy_sample(Cosmo, galaxy_sample, cmb_sample, data_dir, cmb_box, want_random, return_mask=False, return_bias=False, return_tau=False, return_asymm=-1):

    # filename of galaxy map
    if galaxy_sample == "2MPZ":
        gal_fn = data_dir+"/2mpz_data/2MPZ_FULL_wspec_coma_complete.fits"#2MPZ.fits
        mask_fn = data_dir+"/2mpz_data/WISExSCOSmask.fits"
    elif galaxy_sample == "2MPZ_Biteau":
        #gal_fn = data_dir+"/2mpz_data/2MPZ_Biteau.npz"
        #gal_fn = data_dir+"/2mpz_data/2MPZ_Biteau_radec.npz"
        gal_fn = data_dir+"/2mpz_data/2MPZ_Biteau_radec_cut.npz" # remove unidentified objects
        mask_fn = data_dir+"/2mpz_data/WISExSCOSmask.fits"
    elif galaxy_sample == "WISExSCOS":
        gal_fn = data_dir+"wisexscos_data/WIxSC.fits"
        mask_fn = data_dir+"/2mpz_data/WISExSCOSmask.fits"
    elif galaxy_sample == "DECALS":
        gal_fn1 = data_dir+"dels_data/Legacy_Survey_DECALS_galaxies-selection.fits"
        gal_fn2 = data_dir+"dels_data/Legacy_Survey_BASS-MZLS_galaxies-selection.fits"
        mask_fn = data_dir+"dels_data/Legacy_footprint_final_mask_cut_decm36_galactic.fits"
    elif galaxy_sample == "BOSS_North":
        gal_fn = data_dir+"/boss_data/galaxy_DR12v5_CMASS_North.fits"
    elif galaxy_sample == "BOSS_South":
        gal_fn = data_dir+"/boss_data/galaxy_DR12v5_CMASS_South.fits"
    elif galaxy_sample == "eBOSS_SGC":
        gal_fn = data_dir+"/eboss_data/eBOSS_ELG_clustering_data-SGC-vDR16.fits"
    elif galaxy_sample == "eBOSS_NGC":
        gal_fn = data_dir+"/eboss_data/eBOSS_ELG_clustering_data-NGC-vDR16.fits"
    elif "SDSS" in galaxy_sample:
        gal_fn = data_dir+"/sdss_data/V21_DR15_Catalog_v4.txt"
    elif "MGS" == galaxy_sample:
        kcorr_fn = data_dir+"/sdss_data/kcorrect.nearest.petro.z0.10.fits"
        gal_fn = data_dir+"/sdss_data/post_catalog.dr72bright0.fits"
        bias_fn = data_dir+"/sdss_data/bias_dr72bright0.npz"
    elif "MGS_grp" == galaxy_sample:
        gal_fn = data_dir+"/sdss_data/sdss_kdgroups_v1.0.dat"
        #mask_fn = data_dir+"/sdss_data/mask_sdss_kdgroups_v1.0.fits"
        
    # load galaxy sample
    if galaxy_sample == '2MPZ':
        hdul = fits.open(gal_fn)
        RA = hdul[1].data['RA'].flatten()/utils.degree # 0, 360
        DEC = hdul[1].data['DEC'].flatten()/utils.degree # -90, 90
        print("DEC min/max", DEC.min(), DEC.max())
        ZPHOTO = hdul[1].data['ZPHOTO'].flatten()
        ZSPEC = hdul[1].data['ZSPEC'].flatten()
        
        #mode = "ZPHOTO"
        #mode = "ZSPEC" # complete for K < 11.65
        mode = "ZMIX"
        if mode == "ZPHOTO":
            Z = ZPHOTO.copy()
        elif mode == "ZSPEC":
            Z = ZSPEC.copy()
        elif mode == "ZMIX":
            Z = ZPHOTO.copy()
            Z[ZSPEC > 0.] = ZSPEC[ZSPEC > 0.]
        K_rel = hdul[1].data['KCORR'].flatten() # might be unnecessary since 13.9 is the standard
        B = hdul[1].data['B'].flatten() # -90, 90
        L = hdul[1].data['L'].flatten() # 0, 360
        if want_random != -1:
            np.random.seed(want_random)
            factor = 3
            N_rand = len(RA)*factor
            Z = np.repeat(Z, factor)
            K_rel = np.repeat(K_rel, factor)
            """
            # another version
            RA = np.repeat(RA, factor)
            DEC = np.repeat(DEC, factor)
            inds_ra = np.arange(len(RA), dtype=int)
            inds_dec = np.arange(len(RA), dtype=int)
            np.random.shuffle(inds_ra)
            np.random.shuffle(inds_dec)
            RA = RA[inds_ra]
            DEC = DEC[inds_dec]
            """
            costheta = np.random.rand(N_rand)*2.-1.
            theta = np.arccos(costheta)
            DEC = theta*(180./np.pi) # 0, 180
            DEC -= 90.
            RA = np.random.rand(N_rand)*360.
            print("RA/DEC range", RA.min(), RA.max(), DEC.min(), DEC.max())
            c_icrs = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs') # checked
            B = c_icrs.galactic.b.value
            L = c_icrs.galactic.l.value

        #choice = (K_rel < 13.9) & (Z > 0.0) # original is 13.9
        choice = (K_rel < 13.9) & (Z > 0.02) & (Z < 0.35)  # original is 13.9
        #choice = (K_rel < 13.9) & (Z > 0.0) & (Z < 0.0773)
        #choice = (K_rel < 13.9) & (Z > 0.0)
        #choice = (K_rel < 11.65) & (Z > 0.0) & (Z < 0.3)
        #choice = (K_rel < 13.9) & (Z < 0.15); Z[Z < 0.] = 0.

        # make a cut in luminosity
        lum_cut = True
        if lum_cut:
            K_abs = np.zeros_like(K_rel)+100. # make it faint
            for i in range(len(Z)):
                if Z[i] < 0.: continue
                lum_dist = Cosmo.luminosity_distance(Z[i]) # Mpc
                E_z = Z[i]
                K_z = -6.*np.log10(1. + Z[i])
                K_abs[i] = K_rel[i] - 5.*np.log10(lum_dist) - 25. + K_z + E_z
            K_perc = np.percentile(K_abs, 40.) #33.)
            print(K_abs.min(), K_perc, K_abs.max())
            
            choice &= (K_abs < K_perc)
            if return_tau:
                K_abs = K_abs[choice]
                tau = 10.**(0.4*(0.-K_abs))
                
        # apply 2MPZ mask
        B *= utils.degree
        L *= utils.degree
        x = np.cos(B)*np.cos(L)
        y = np.cos(B)*np.sin(L)
        z = np.sin(B)
        mask = hp.read_map(mask_fn) # ring, not nested
        npix = len(mask)
        nside = hp.npix2nside(npix)
        ipix = hp.pixelfunc.vec2pix(nside, x, y, z)
        choice &= mask[ipix] == 1.
        
    elif '2MPZ_Biteau' == galaxy_sample:
        data = np.load(gal_fn)
        RA = data['RA']
        DEC = data['DEC']
        Z = data['Z']
        #Z = data['Z_hdul'] # redshifts coming from original data
        Mstar = data['M_star']
        d_L = data['d_L']
        B = data['B']
        L = data['L']
        choice = np.ones(len(Z), dtype=bool)

        c_icrs = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs') # checked
        B = c_icrs.galactic.b.value
        L = c_icrs.galactic.l.value

        # apply 2MPZ mask
        B *= utils.degree
        L *= utils.degree
        x = np.cos(B)*np.cos(L)
        y = np.cos(B)*np.sin(L)
        z = np.sin(B)
        mask = hp.read_map(mask_fn) # ring, not nested
        npix = len(mask)
        nside = hp.npix2nside(npix)
        ipix = hp.pixelfunc.vec2pix(nside, x, y, z)
        choice &= mask[ipix] == 1.
        
        mass_cut = True
        if mass_cut:
            Mstar_perc = np.percentile(Mstar, 30.)
            print("Mstar threshold = ", Mstar_perc)
            Mstar_perc = 10.3 
            print("Mstar threshold = ", Mstar_perc)
            choice &= (Mstar > Mstar_perc)
        #dL_max = 350. # Mpc sample complete 0.0773
        dL_max = np.max(d_L) # Mpc sample complete 0.0773
        #dL_min = 100. # Mpc 0.0229
        dL_min = 0. # Mpc 0.
        print("d_L min/max = ", dL_min, dL_max)
        choice &= (d_L < dL_max) & (d_L > dL_min)
        
    elif galaxy_sample == 'WISExSCOS':
        hdul = fits.open(gal_fn)
        RA = hdul[1].data['RA'].flatten()/utils.degree # 0, 360
        DEC = hdul[1].data['DEC'].flatten()/utils.degree # -90, 90
        B = hdul[1].data['B'].flatten() # 0, 360
        L = hdul[1].data['L'].flatten() # -90, 90
        print("DEC min/max", DEC.min(), DEC.max())
        Z = hdul[1].data['ZPHOTO_CORR'].flatten()
        choice = (Z > 0.1) & (Z < 0.35)
        
        # apply mask
        B *= utils.degree
        L *= utils.degree
        x = np.cos(B)*np.cos(L)
        y = np.cos(B)*np.sin(L)
        z = np.sin(B)
        mask = hp.read_map(mask_fn) # ring, not nested
        npix = len(mask)
        nside = hp.npix2nside(npix)
        ipix = hp.pixelfunc.vec2pix(nside, x, y, z)
        choice &= mask[ipix] == 1.

    elif galaxy_sample == "DECALS":
        hdul = fits.open(gal_fn1)
        RA = hdul[1].data['RA'].flatten() # 0, 360
        DEC = hdul[1].data['DEC'].flatten() # -90, 90
        #Z = hdul[1].data['PHOTOZ_3DINFER'].flatten() # many negative - bad signal
        Z = hdul[1].data['PHOTOZ_ZHOU'].flatten() # few negative - better signal
        
        hdul = fits.open(gal_fn2)
        RA = np.hstack((RA, hdul[1].data['RA'].flatten())) # 0, 360
        DEC = np.hstack((DEC, hdul[1].data['DEC'].flatten())) # -90, 90
        #Z = np.hstack((Z, hdul[1].data['PHOTOZ_3DINFER'].flatten()))
        Z = np.hstack((Z, hdul[1].data['PHOTOZ_ZHOU'].flatten()))
        
        choice = np.ones(len(Z), dtype=bool) # DELS__1 (all galaxies)
        #choice = (Z > 0.0) & (Z < 0.8) # DELS__0 if 3DINFER
        #choice = (Z > 0.)
        
        c_icrs = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs') # checked
        B = c_icrs.galactic.b.value
        L = c_icrs.galactic.l.value

        # apply mask
        B *= utils.degree
        L *= utils.degree
        x = np.cos(B)*np.cos(L)
        y = np.cos(B)*np.sin(L)
        z = np.sin(B)
        mask = hp.read_map(mask_fn) # ring, not nested
        npix = len(mask)
        nside = hp.npix2nside(npix)
        ipix = hp.pixelfunc.vec2pix(nside, x, y, z)
        choice &= mask[ipix] == 1.
        print("DEC min/max", DEC.min(), DEC.max())
        print("RA min/max", RA.min(), RA.max())

    elif 'BOSS' in galaxy_sample:
        hdul = fits.open(gal_fn)
        RA = hdul[1].data['RA'].flatten() # 0, 360
        DEC = hdul[1].data['DEC'].flatten() # -90, 90 # -10, 36
        Z = hdul[1].data['Z'].flatten()
        choice = np.ones(len(Z), dtype=bool)
        
    elif 'eBOSS' in galaxy_sample:
        hdul = fits.open(gal_fn)
        RA = hdul[1].data['RA'].flatten() # 0, 360
        DEC = hdul[1].data['DEC'].flatten() # -90, 90 # -10, 36
        Z = hdul[1].data['Z'].flatten()
        choice = np.ones(len(Z), dtype=bool)
        
    elif 'SDSS' in galaxy_sample:
        data = ascii.read(gal_fn)
        RA = data['ra'] # 0, 360
        DEC = data['dec'] # -90, 90 # -10, 36
        Z = data['z']
        lum = data['lum']
        stellarmasses = data['stellarmasses']
        halomasses = data['halomasses']

        # test if one or two samples
        splits = galaxy_sample.split('_')
        L_lo, L_hi = get_sdss_lum_lims(galaxy_sample)
        if return_asymm >= 0:
            if len(splits) == 2:
                L_lo_asymm, L_hi_asymm = get_sdss_lum_lims('_'.join([splits[0], splits[1]]))
            else:
                L_lo_asymm, L_hi_asymm = get_sdss_lum_lims('_'.join([splits[0], splits[2]]))
        choice = (L_lo < lum) & (L_hi >= lum)
        if "ACT_BN" in cmb_sample or "ACT_D56" in cmb_sample:
            choice &= data['S16ILC'] == 1.
        elif "ACT_DR5_f090" in cmb_sample or "ACT_DR5_f150" in cmb_sample:
            choice &= data['S18coadd'] == 1.
        print("galaxies satisfying luminosity cut = ", np.sum(choice))

        if return_asymm >= 0:
            choice_asymm = (L_lo_asymm < lum) & (L_hi_asymm >= lum)
            if "ACT_BN" in cmb_sample or "ACT_D56" in cmb_sample:
                choice_asymm &= data['S16ILC'] == 1.
            elif "ACT_DR5_f090" in cmb_sample or "ACT_DR5_f150" in cmb_sample:
                choice_asymm &= data['S18coadd'] == 1.
            
        # read in halo mass and stellar mass and convert into optical depth
        if return_tau:
            tau = get_tau_theory(stellarmasses, halomasses)
        
        # get bias estimate
        if return_bias:
            bias = get_bias(halomasses, Z) # from empirical estimate

    elif 'MGS' == galaxy_sample:
        hdul = fits.open(gal_fn)
        RA = hdul[1].data['RA'].flatten() # 0, 360
        DEC = hdul[1].data['DEC'].flatten() # -90, 90 # -10, 70
        Z = hdul[1].data['Z'].flatten()
        ABSM = hdul[1].data['ABSM'][:, 2].flatten() # ugrizJKY (r)
        M = hdul[1].data['M'].flatten() # ugrizJKY (r)
        OBJECT_POSITION = hdul[1].data['OBJECT_POSITION'].flatten()
        hdul = fits.open(kcorr_fn)
        KCORRECT = hdul[1].data['KCORRECT']

        # get luminosity given magnitude
        lum = get_lum(Z, M, OBJECT_POSITION, KCORRECT)
        print("hopefully not many L > 1.e11 = ", np.sum(lum > 1.e11))
        lum[lum > 1.e11] = 1.e11

        # get halo mass estimate
        M_of_L_cen = get_M_of_L_cen()
        halo_mass = M_of_L_cen(lum) # Msun/h
        halo_mass[halo_mass > 3.e14] = 3.e14
        halo_mass[halo_mass > 1.e11] = 1.e11

        # get theoretical optical depth estimate
        h = 0.6774
        tau = get_tau_theory(3.*lum/h**2, halo_mass/h)
        
        # get bias estimate
        if return_bias:
            #bias = np.load(bias_fn)['bias'] # auto-correlation
            #bias = get_bias(halo_mass/h, Z) # colossus
            bias = get_bias_mgs(lum) # empirical estimate Zehavi 2011
        
        # make selection based on luminosity, magnitude, halo mass
        #choice = ABSM < -21. # original try 160k
        #choice = ABSM < -21.2 # original try 120k
        choice = ABSM < -21.4 # original try 86k nice shape sometimes 3 sigma with 3
        #choice = lum >= np.percentile(lum, 75) # median was upside down
        #choice = ABSM < -21.6 # 60k 3.1 gives 3 sigma with 100 samples
        #choice = ABSM < -21.7 # 50k
        #choice = ABSM < -21.8 # 40k
        #choice &= (Z > 0.09)
        #choice = np.ones(len(Z), dtype=bool)

    elif 'MGS_grp' in galaxy_sample:
        data = ascii.read(gal_fn)
        RA = data['col2'] # 0, 360
        DEC = data['col3'] # -90, 90 # -10, 70
        Z = data['col4']
        Lgal = data['col5']
        Psat = data['col7']
        Mgrp = data['col8']
        Ltot = data['col10']

        # get theoretical optical depth estimate
        h = 0.6774
        tau = get_tau_theory(3.*Ltot/h**2, Mgrp/h)
        
        # get bias estimate
        if return_bias:
            bias = get_bias(Mgrp, Z) # colossus
        
        # make selection based on luminosity, magnitude, halo mass
        choice = Psat < 0.5
        #choice &= Mgrp > 2.e13
        choice &= Mgrp > 8.e12
        if return_asymm >= 0:
            choice_asymm = Psat < 0.5
        
    # galaxy indices before applying any cuts
    index = np.arange(len(Z), dtype=int)
    print("Zmin/max/med = ", Z.min(), Z.max(), np.median(Z))
    print("RAmin/RAmax = ", RA.min(), RA.max())
    print("DECmin/DECmax = ", DEC.min(), DEC.max())

    # compute comoving position and angular distance
    P, D_A = get_P_D_A(Cosmo, RA, DEC, Z)
    
    # make magnitude and RA/DEC cuts to  match ACT
    DEC_choice = (DEC <= cmb_box['decto']) & (DEC > cmb_box['decfrom'])
    if cmb_sample == 'ACT_D56':
        RA_choice = (RA <= cmb_box['rafrom']) | (RA > cmb_box['rato'])
    elif cmb_sample == 'ACT_BN':
        RA_choice = (RA <= cmb_box['rafrom']) & (RA > cmb_box['rato'])
    else:
        # DR5 has RA from -180 to 180, so cmb_box is not used
        RA_choice = np.ones_like(DEC_choice)
    RADEC_choice = DEC_choice & RA_choice

    # second sample to cross-correlate with
    if return_asymm == 1:
        choice_asymm &= RADEC_choice
        RA_asymm = RA[choice_asymm]
        DEC_asymm = DEC[choice_asymm]
        Z_asymm = Z[choice_asymm]
        P_asymm = P[choice_asymm]
        D_A_asymm = D_A[choice_asymm]
        index_asymm = index[choice_asymm]
        if return_bias:
            bias_asymm = bias[choice_asymm]
            print("median bias = ", np.median(bias_asymm))
            print("mean bias = ", np.mean(bias_asymm))
            quit()
        else:
            print("bias not wanted, returning ones")
            bias_asymm = np.ones_like(RA_asymm)
    
    # apply RA, DEC cuts
    choice &= RADEC_choice
    RA = RA[choice]
    DEC = DEC[choice]
    Z = Z[choice]
    P = P[choice]
    D_A = D_A[choice]
    index = index[choice]
    
    # if second sample is same as the first sample
    if return_asymm == 0:
        RA_asymm, DEC_asymm, Z_asymm, P_asymm, D_A_asymm, index_asymm = RA, DEC, Z, P, D_A, index 
        if return_bias:
            bias_asymm = bias[choice]
            print("median bias = ", np.median(bias_asymm))
            print("mean bias = ", np.mean(bias_asymm))            
        else:
            print("bias not wanted, returning ones")            
            bias_asymm = np.ones_like(RA_asymm)

    # optical depth estimate
    if return_tau:
        tau = tau[choice]
        print("median tau = ", np.median(tau))
        print("mean tau = ", np.mean(tau))
    else:
        print("tau not wanted, returning ones")
        tau = np.ones_like(RA)
    print("number of galaxies = ", np.sum(choice))
        
    # just to see the numbers
    if galaxy_sample == "2MPZ":
        if mode == "ZMIX":
            ZSPEC = ZSPEC[choice]
            assert len(ZSPEC) == len(Z)
            print("percentage zspec available = ", np.sum(ZSPEC > 0.)*100./len(ZSPEC))
        
    # formula for tau divides by d_A^2
    if return_tau:
        #tau /= D_A**2 # because we set D_A = 1 Mpc
        pass
    # fields we are outputting
    result = [RA, DEC, Z, P, D_A, index, tau]
    if return_asymm >= 0:
        print("number of galaxies for cross-correlation = ", len(RA_asymm))
        result += [RA_asymm, DEC_asymm, Z_asymm, P_asymm, D_A_asymm, index_asymm, bias_asymm]
    if return_mask:
        result.append(mask)
    return result

def get_sdss_lum_lims(galaxy_sample):
    if "all" in galaxy_sample:
        L_lo = 0.
        L_hi = 1.e20
    if "L43" in galaxy_sample:
        L_lo = 4.3e10
        if "L43D" in galaxy_sample:
            L_hi = 6.1e10
        else:
            L_hi = 1.e20
    elif "L61" in galaxy_sample:
        L_lo = 6.1e10
        if "L61D" in galaxy_sample:
            L_hi = 7.9e10
        else:
            L_hi = 1.e20
    elif "L79" in galaxy_sample:
        L_lo = 7.9e10
        L_hi = 1.e20
    return L_lo, L_hi

def get_lum(Z, M, OBJECT_POSITION, KCORRECT):

    # In the following, I will use absolute magnitudes calculated assuming Planck13
    distmod = Planck13.distmod(Z).value
    k = KCORRECT[:, 2][OBJECT_POSITION]
    m_abs = (M - k - distmod - 5 * np.log10(Planck13.H0.value / 100.0) + 0.010)

    # We also apply the evolution correction (http://cosmo.nyu.edu/blanton/vagc/lss.html)
    q0 = 2.0; q1 = -1.0
    m_abs = (m_abs + q0 * (1 + q1 * (Z - 0.1)) * (Z - 0.1))

    # Calculate luminosities.
    lum = 10**(-0.4 * (m_abs - 4.76))
    return lum

def get_bias_mgs(L, b_star=1.14, L_star=1.20e10):
    # L∗ = 1.20 × 10^10 h^-2 L⊙ in the r-band; Blanton et al. 2003c
    # bg/b∗ = 0.85 + 0.15L/L∗ − 0.04(M − M∗)
    b_g = b_star*(0.85 + 0.15*L/L_star)
    return b_g

def get_L_cen(M_h, A=0.32, M_t=3.08e11, alpha_M=0.264, L_star=1.2e10):
    # units of L are h^-2 Lodot; M are Modot/h
    L_cen = L_star*A*(M_h/M_t)**alpha_M*np.exp(-M_t/M_h+1.)
    return L_cen

def get_M_of_L_cen(plot=False):
    """ get halo mass a function of central r-band luminosity """
    M = np.geomspace(1.e10, 3.e15, 10000) # Modot/h
    L_cen = get_L_cen(M)
    M_of_L_cen = interp1d(L_cen, M)
    
    if plot:
        plt.figure(figsize=(9, 7))
        plt.plot(M, L_cen)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim([M.min(), M.max()])
        plt.show()
    return M_of_L_cen

def get_bias(Mvirs, redshifts):
    """ get bias as a function of halo mass (units Modot/h) and redshift """
    #nu = peaks.peakHeight(M, z)
    #b = bias.haloBiasFromNu(nu, model = 'sheth01')
    b = bias.haloBias(Mvirs, model='tinker10', z=redshifts, mdef='200m')#mdef='vir')
    return b

def get_tau_theory(M_star, M_vir, d_A=1.):
    # M_vir within 2.1 arcmin in units of Modot; d_A in Mpc
    f_b = 0.157 # Omega_b/Omega_m
    f_star = M_star/M_vir
    x_e = (X_H + 1.)/(2.*X_H)
    tau = sigma_T_over_m_p * solar_mass_over_Mpc_to_cm_squared * x_e * X_H * (1-f_star) * f_b * M_vir / d_A**2
    return tau
