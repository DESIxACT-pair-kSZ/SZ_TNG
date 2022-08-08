import numpy as np
import numba
from numba import njit
from nbodykit.lab import ArrayCatalog, FieldMesh
#from nbodykit.base.mesh import MeshFilter

from pixell import enmap, enplot, utils
import rotfuncs


def cutoutGeometry(projCutout='cea', rApMaxArcmin=6., resCutoutArcmin=0.25, test=False):
    '''Create enmap for the cutouts to be extracted.
    Returns a null enmap object with the right shape and wcs.
    '''

    # choose postage stamp size to fit the largest ring
    dArcmin = np.ceil(2. * rApMaxArcmin * np.sqrt(2.))
   
    nx = np.floor((dArcmin / resCutoutArcmin - 1.) / 2.) + 1.
    dxDeg = (2. * nx + 1.) * resCutoutArcmin / 60.
    ny = np.floor((dArcmin / resCutoutArcmin - 1.) / 2.) + 1.
    dyDeg = (2. * ny + 1.) * resCutoutArcmin / 60.

    # define geometry of small square maps to be extracted
    shape, wcs = enmap.geometry(np.array([[-0.5*dxDeg,-0.5*dyDeg],[0.5*dxDeg,0.5*dyDeg]])*utils.degree, res=resCutoutArcmin*utils.arcmin, proj=projCutout)
    cutoutMap = enmap.zeros(shape, wcs)
    
    if test:
        print("cutout sides are dx, dy =", dxDeg*60., ",", dyDeg*60. , "arcmin")
        print("cutout pixel dimensions are", shape)
        print("hence a cutout resolution of", dxDeg*60./shape[0], ",", dyDeg*60./shape[1], "arcmin per pixel")
        print("(requested", resCutoutArcmin, "arcmin per pixel)")

    return cutoutMap

def extractStamp(cmbMap, ra, dec, rApMaxArcmin, resCutoutArcmin, projCutout, pathTestFig='figs/', test=False, cmbMask=None, order=1):
    """Extracts a small CEA or CAR map around the given position, with the given angular size and resolution.
    ra, dec in degrees.
    Does it for the map, the mask and the hit count.
    order > 1 is too slow
    """
    # enmap 
    stampMap = cutoutGeometry(rApMaxArcmin=rApMaxArcmin, resCutoutArcmin=resCutoutArcmin, projCutout=projCutout)
    stampMask = stampMap.copy()
    
    # coordinates of the square map (between -1 and 1 deg); output map position [{dec,ra},ny,nx]
    opos = stampMap.posmap()

    # coordinate of the center of the square map we want to extract
    sourcecoord = np.array([ra, dec])*utils.degree  # convert from degrees to radians

    # corresponding true coordinates on the big healpy map
    ipos = rotfuncs.recenter(opos[::-1], [0, 0, sourcecoord[0], sourcecoord[1]])[::-1]

    # Here, I use bilinear interpolation
    stampMap[:, :] = cmbMap.at(ipos, prefilter=True, mask_nan=False, order=order)
    if cmbMask is not None:
        stampMask[:, :] = cmbMask.at(ipos, prefilter=True, mask_nan=False, order=order)
    
        # re-threshold the mask map, to keep 0 and 1 only
        stampMask[:, :] = 1.*(stampMask[:, :]>0.5)
    
    if test:
        print("Extracted cutouts around ra=", ra, "dec=", dec)
        print("- min, mean, max =", np.min(stampMap), np.mean(stampMap), np.max(stampMap))
        # don't save if empty
        if np.min(stampMap) + np.mean(stampMap) + np.max(stampMap) == 0.: return opos, stampMap

        plots = enplot.plot(enmap.upgrade(stampMap, 5), grid=True)
        enplot.write(pathTestFig+"/stampmap_ra"+str(np.round(ra, 2))+"_dec"+str(np.round(dec, 2)), plots)

    if cmbMask is not None:
        return opos, stampMap, stampMask
    return opos, stampMap

def calc_T_AP(imap, rad_arcmin, test=False, mask=None, divmap=None):
    modrmap = imap.modrmap()
    radius = rad_arcmin*utils.arcmin
    inner = modrmap < radius
    outer = (modrmap >= radius) & (modrmap < np.sqrt(2.)*radius)
    if mask is None:
        flux_inner = imap[inner].mean()
        flux_outer = imap[outer].mean()
        flux_inner_std = imap[inner].std()
        flux_outer_std = imap[outer].std()
        if divmap is not None:
            divs = divmap[inner].mean()
    else:
        if (np.sum(mask[inner]) == 0.) or (np.sum(mask[outer]) == 0.):
            return 0.
        else:
            flux_inner = np.sum(imap[inner]*mask[inner])/np.sum(mask[inner])
            flux_outer = np.sum(imap[outer]*mask[outer])/np.sum(mask[outer])
            flux_inner_std = np.sqrt(np.sum((imap[inner]*mask[inner]-flux_inner)**2)/np.sum(mask[inner]))
            flux_outer_std = np.sqrt(np.sum((imap[outer]*mask[outer]-flux_outer)**2)/np.sum(mask[outer]))
            if divmap is not None:
                divs = np.sum(divmap[inner]*mask[inner])/np.sum(mask[inner])
    
    if divmap is not None:
        flux_inner, flux_outer, flux_inner_std, flux_outer_std, divs
    return flux_inner, flux_outer, flux_inner_std, flux_outer_std
