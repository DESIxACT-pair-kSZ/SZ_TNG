# (((1/1+z=1)*500)^2/1150^2*(180/pi)^2 
import numpy as np
import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

def get_smooth_density(D, fwhm, Lbox, N_dim):
    #karr = np.fft.fftfreq(N_dim, d=Lbox/(2*np.pi*N_dim))
    kstep = 2*np.pi/(N_dim*np.pi/180*pixsizedeg)
    karr = np.fft.fftfreq(N_dim, d=Lboxdeg*np.pi/180./(2*np.pi*N_dim))
    print("kstep = ", kstep, karr[1]-karr[0]) # N_dim/d gives kstep
    dfour = np.fft.fftn(D)
    dksmo = np.zeros((N_dim, N_dim),dtype=complex)
    ksq = np.zeros((N_dim, N_dim),dtype=complex)
    ksq[:, :] = karr[None, :]**2+karr[:,None]**2
    dksmo[:, :] = gauss_beam(ksq, fwhm)*dfour
    drsmo = np.real(np.fft.ifftn(dksmo))                                                                                                                                                                                     
    return drsmo 

def gauss_beam(ellsq, fwhm):
    tht_fwhm = np.deg2rad(fwhm/60.)
    return np.exp(-(tht_fwhm**2.)*(ellsq)/(8.*np.log(2.)))

dirs = ['xy', 'yz', 'zx']
N_dim = 10000
fwhm = 1.3 #arcmin 2009.05557 not ideal
fwhm = 2.1 #arcmin 2009.05557
Lbox = 500.

# angular distance
redshift = 1.
a = 1./(1+redshift)
h = 0.6774
cosmo = FlatLambdaCDM(H0=h*100., Om0=0.3089, Tcmb0=2.725)
d_L = cosmo.luminosity_distance(redshift).to(u.Mpc).value
d_A = d_L/(1.+redshift)**2 # dA = dL/(1+z)^2 # Mpc
d_A *= h
print("d_A =", d_A) # Mpc/h
Lboxdeg = np.sqrt((a*Lbox)**2/d_A**2*(180./np.pi)**2)
pixsizedeg = Lboxdeg/N_dim
print("Lboxdeg = ", Lboxdeg)

for i in range(len(dirs)):
    
    Y_xy = np.load(f"data_maps_sz/Y_compton_{dirs[i]:s}.npy")

    Y_beam_xy = get_smooth_density(Y_xy, fwhm, Lbox, N_dim)
    std_Y_xy = Y_beam_xy.std()
    mean_Y_xy = Y_beam_xy.mean()

    plt.figure(figsize=(16,14))
    plt.imshow(np.log10(1+Y_xy), vmin=np.log10(1+mean_Y_xy-4.*std_Y_xy), vmax=np.log10(1+mean_Y_xy+4.*std_Y_xy))
    plt.savefig(f"figs/Y_{dirs[i]:s}.png")
    
    plt.figure(figsize=(16,14))
    plt.imshow(np.log10(1+Y_beam_xy), vmin=np.log10(1+mean_Y_xy-4.*std_Y_xy), vmax=np.log10(1+mean_Y_xy+4.*std_Y_xy))
    plt.savefig(f"figs/Y_beam_{dirs[i]:s}.png")

    b_xy = np.load(f"data_maps_sz/b_{dirs[i]:s}.npy")

    b_beam_xy = get_smooth_density(b_xy, fwhm, Lbox, N_dim)
    std_b_xy = b_beam_xy.std()
    mean_b_xy = b_beam_xy.mean() 

    plt.figure(figsize=(16,14))
    plt.imshow((1+b_xy), vmin=(1+mean_b_xy-4.*std_b_xy), vmax=(1+mean_b_xy+4.*std_b_xy))
    plt.savefig(f"figs/b_{dirs[i]:s}.png")
    
    plt.figure(figsize=(16,14))
    plt.imshow((1+b_beam_xy), vmin=(1+mean_b_xy-4.*std_b_xy), vmax=(1+mean_b_xy+4.*std_b_xy))
    plt.savefig(f"figs/b_beam_{dirs[i]:s}.png")

