"""
Script for obtaining beamed SZ maps
Size of box in degrees: (((1/1+z=1)*Lboxmpch)^2/d_Ampch^2*(180/pi)^2 
"""
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 30})
import matplotlib.pyplot as plt
import h5py
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import argparse

def get_smooth_density(D, fwhm, Lbox, N_dim, pixsizedeg):
    # if using physical karr, be mindful of Lbox units
    """
    Smooth density map D ((0, Lbox] and N_dim^2 cells) with Gaussian beam of FWHM
    """
    kstep = 2*np.pi/(N_dim*np.pi/180*pixsizedeg)
    #karr = np.fft.fftfreq(N_dim, d=Lbox/(2*np.pi*N_dim)) # physical (not correct)
    Lboxdeg = pixsizedeg*N_dim
    karr = np.fft.fftfreq(N_dim, d=Lboxdeg*np.pi/180./(2*np.pi*N_dim)) # angular
    print("kstep = ", np.round(kstep,2), np.round(karr[1]-karr[0],2)) # N_dim/d gives kstep

    # fourier transform the map and apply gaussian beam
    dfour = np.fft.fftn(D)
    dksmo = np.zeros((N_dim, N_dim), dtype=complex)
    ksq = np.zeros((N_dim, N_dim), dtype=complex)
    ksq[:, :] = karr[None, :]**2+karr[:,None]**2
    dksmo[:, :] = gauss_beam(ksq, fwhm)*dfour
    drsmo = np.real(np.fft.ifftn(dksmo))

    return drsmo 

def gauss_beam(ellsq, fwhm):
    """
    Gaussian beam of size fwhm
    """
    tht_fwhm = np.deg2rad(fwhm/60.)
    return np.exp(-(tht_fwhm**2.)*(ellsq)/(8.*np.log(2.)))
    
# returns beamed Y array if savemaps == False
def main(args):
    LHnum = args.LHnum
    snapinit = args.snapinit
    snapfinal = args.snapfinal
    nbins = args.nbins
    savemaps = args.savemaps
    # check to see that LH and snap nums are valid
    LHlist = [str(i) for i in range(1000)]
    snaplist = np.arange(34)

    if LHnum not in LHlist:
        ('LH number out of bounds')
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

    # simulation choices
    basePath = "/global/cscratch1/sd/kjc268/CAMELS_ITNG/LH_{}/snap_{}.hdf5".format(LHnum,str(snapinit).zfill(3))
    n_chunks = h5py.File(basePath, "r")['Header'].attrs['NumFilesPerSnapshot']
    z = h5py.File(basePath, "r")['Header'].attrs['Redshift'] # TNG
    a = 1./(1+z)
    c = 29979245800. # cm/s
    Omega0= h5py.File(basePath, "r")['Header'].attrs['Omega0']
    h = h5py.File(basePath, "r")['Header'].attrs['HubbleParam']

    Lbox = 25. # cMpc/h 
    dirs = ['xy', 'yz', 'zx']
    N_dim = nbins-1

    #cosmo stuff 
    paramsfile = "/global/cscratch1/sd/kjc268/CAMELS_ITNG/LH_{}/CosmoAstro_params.txt".format(LHnum)
    f = open(paramsfile,"r")
    params = f.read().split(" ")
    omegam = float(params[0])
    sigma8 = float(params[1])
    #Asn1 = float(params[2]);Asn2 = float(params[3]);Aagn1 = float(params[4]);Aagn2 = float(params[5]) 
    f.close()

    # gaussian beam
    #fwhm = 1.3 #arcmin 2009.05557 not ideal
    fwhm = 2.1 #arcmin 2009.05557

    # compute angular distance
    cosmo = FlatLambdaCDM(H0=h*100., Om0=Omega0, Tcmb0=2.725)
    d_L = cosmo.luminosity_distance(z).to(u.Mpc).value
    d_A = d_L/(1.+z)**2 # dA = dL/(1+z)^2 # Mpc
    d_A *= h 
    print("d_A =", np.round(d_A,2), " Mpc/h") # Mpc/h

    # box size in degrees
    Lboxdeg = np.sqrt((a*Lbox)**2/d_A**2*(180./np.pi)**2)
    pixsizedeg = Lboxdeg/N_dim
    print("Lboxdeg = ", np.round(Lboxdeg,2))

    if snapfinal == snapinit: 
        snapshot = str(snapinit).zfill(3)
    else: 
        snapshot = f'{str(snapinit).zfill(3):s}_to_{str(snapfinal).zfill(3):s}_lc'
   
    Ys = []
    
    # for each of three projections
    for i in range(len(dirs)):

        # load tSZ map
        Y_xy = np.load(f"{save_dir}/Y_compton_{dirs[i]:s}_snap_{snapshot:s}.npy")
        # smooth with beam
        Y_beam_xy = get_smooth_density(Y_xy, fwhm, Lbox, N_dim, pixsizedeg)
        std_Y_xy = Y_beam_xy.std()
        mean_Y_xy = Y_beam_xy.mean()
        
        if savemaps != True:
            Ys.append(Y_beam_xy)

        else:
            # plot and save unbeamed
            fig, ax = plt.subplots(1, 1, figsize=(16,14))
            ax.set_xlabel('cMpc/h')
            ax.set_ylabel('cMpc/h')
            ticks = [str(np.round(x))[:-2] for x in np.linspace(0,Lbox,11)]
            ax.set_xticks(np.arange(N_dim, 0, step=-N_dim/10))
            ax.set_xticklabels(ticks[0:-1])
            ax.set_yticks(np.arange(0, N_dim, step=N_dim/10))
            ax.set_yticklabels(ticks[0:-1])
            ax.set_title("z = "+"{:.2e}".format(z)+ " $\Omega_m$={}".format(omegam)+ " $\sigma_8$={}".format(sigma8))
            # set zero values to min of array
            mask = np.ma.array(Y_xy, mask=Y_xy<=0).min()
            newY= np.where(Y_xy == 0, mask, Y_xy)
            im = plt.imshow(np.log10(newY),cmap='inferno')
            # vmin=np.log10(1+mean_Y_xy-4.*std_Y_xy), vmax=np.log10(1+mean_Y_xy+4.*std_Y_xy),
            cbar = plt.colorbar(im,fraction=0.046, pad=0.04)
            cbar.set_label(r'log $Y_{{{}}}$'.format(dirs[i]))
            plt.savefig(f"{save_dir}/figs/LH{LHnum}_snap_{snapshot}_Y_{dirs[i]:s}.png")
            plt.close()

            # plot and save beamed
            fig, ax = plt.subplots(1, 1, figsize=(16,14))
            ax.set_xlabel('cMpc/h')
            ax.set_ylabel('cMpc/h')
            ticks = [str(np.round(x))[:-2] for x in np.linspace(0,Lbox,11)]
            ax.set_xticks(np.arange(N_dim, 0, step=-N_dim/10))
            ax.set_xticklabels(ticks[0:-1])
            ax.set_yticks(np.arange(0, N_dim, step=N_dim/10))
            ax.set_yticklabels(ticks[0:-1])
            ax.set_title("z = "+"{:.2e}".format(z)+ " $\Omega_m$={}".format(omegam)+ " $\sigma_8$={}".format(sigma8))
            # set zero values to min of array
            mask = np.ma.array(Y_beam_xy, mask=Y_beam_xy<=0).min()
            newY= np.where(Y_beam_xy == 0, mask, Y_beam_xy)
            im = plt.imshow(np.log10(newY),cmap='inferno')
            #vmin=np.log10(1+mean_Y_xy-4.*std_Y_xy), vmax=np.log10(1+mean_Y_xy+4.*std_Y_xy),
            cbar = plt.colorbar(im,fraction=0.046, pad=0.04)
            cbar.set_label(r'log $Yb_{{{}}}$'.format(dirs[i]))
            plt.savefig(f"{save_dir}/figs/LH{LHnum}_snap_{snapshot}_Y_beam_{dirs[i]:s}.png")
            plt.close()
    
    return np.array(Ys)

    #     # load kSZ map
    #     b_xy = np.load(f"{save_dir}/b_{dirs[i]:s}_snap_{snapshot:d}.npy")

    #     # smooth with beam
    #     b_beam_xy = get_smooth_density(b_xy, fwhm, Lbox, N_dim)
    #     std_b_xy = b_beam_xy.std()
    #     mean_b_xy = b_beam_xy.mean() 

    #     # plot and save unbeamed
    #     plt.figure(figsize=(16,14))
    #     im = plt.imshow((1+b_xy), vmin=(1+mean_b_xy-4.*std_b_xy), vmax=(1+mean_b_xy+4.*std_b_xy))
    #     bar = plt.colorbar(im)
    #     plt.savefig(f"{save_dir}/figs/b_{dirs[i]:s}.png")
    #     plt.close()

    #     # plot and save beamed
    #     plt.figure(figsize=(16,14))
    #     im = plt.imshow((1+b_beam_xy), vmin=(1+mean_b_xy-4.*std_b_xy), vmax=(1+mean_b_xy+4.*std_b_xy))
    #     bar = plt.colorbar(im)
    #     plt.savefig(f"{save_dir}/figs/b_beam_{dirs[i]:s}.png")
    #     plt.close()
    
if __name__ == "__main__":

    # command line arguments -- latin hypercube number, initial and final snapshot number, number of bins for maps 
    # note: final snapshot number is the highest redshift for lightcone
    #ex. apply_beam.py -lh 0 -si 32 -sf 31 -nb 10001 -sm True

    parser = argparse.ArgumentParser()
    parser.add_argument('--LHnum','-lh', type=str)
    parser.add_argument('--snapinit','-si', type=int)
    parser.add_argument('--snapfinal','-sf', type=int)
    parser.add_argument('--nbins','-nb', type=int)
    parser.add_argument('--savemaps','-sm',type=bool)
    args = parser.parse_args()
    main(args)