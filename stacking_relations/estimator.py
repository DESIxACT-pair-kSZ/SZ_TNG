# Important note pairwise_velocity_sky and pairwise_momentum agree up to a sign and differ by ~2 in performance
import numba
numba.config.THREADING_LAYER = 'safe'
import numpy as np
import Corrfunc

@numba.njit(parallel=True, fastmath=True)
def pairwise_momentum(X, dT, bins, is_log_bin, dtype=np.float32, nthread=1):
    """
    https://journals.aps.org/prd/pdf/10.1103/PhysRevD.104.043502
    The formula is:
    v_1,2 = 2 Sum_A,B (s_A-s_B) p_AB / (Sum_A,B p_AB^2)
    p_AB = \hat r . (\hat r_A + \hat r_B)
    \vec r = \vec r_A - \vec r_B
    s_A = \hat r_A v_A
    V_los is independently measured
    Does not use periodic boundary conditions
    """

    # multithreading
    numba.set_num_threads(nthread)

    # declare some stuff
    N = len(X)
    zero = dtype(0.)
    half = dtype(0.5)
    one = dtype(1.)
    two = dtype(2.)

    # are the bins linear or logarithmic
    if is_log_bin:
        dbin = dtype(bins[1]/bins[0])
    else:
        dbin = dtype(bins[1]-bins[0])
    fbin = dtype(bins[0])

    # initialize arrays
    pair_count = np.zeros((nthread, len(bins)-1), dtype=dtype)
    weight_count = np.zeros((nthread, len(bins)-1), dtype=dtype)
    norm_weight_count = np.zeros((nthread, len(bins)-1), dtype=dtype)
    pairwise_velocity = np.zeros(len(bins)-1, dtype=dtype)

    # loop over the galaxies 
    for i in numba.prange(N):
        # direction in the sky of the galaxy
        x1, y1, z1 = X[i][0], X[i][1], X[i][2]
        dist1 = np.sqrt(x1**two + y1**two + z1**two)
        if dist1 > zero:
            hx1, hy1, hz1 = x1/dist1, y1/dist1, z1/dist1
        else:
            hx1, hy1, hz1 = zero, zero, zero
            
        # the cluster velocity proxy of the galaxy
        dT1 = dT[i]

        # the id of that thread
        t = numba.np.ufunc.parallel._get_thread_id()
        
        for j in range(N):
            if j <= i: continue # operation is symmetric so save some time

            # direction in the sky of other galaxy
            x2, y2, z2 = X[j][0], X[j][1], X[j][2]
            dist2 = np.sqrt(x2**two + y2**two + z2**two)
            if dist2 > zero:
                hx2, hy2, hz2 = x2/dist2, y2/dist2, z2/dist2
            else:
                hx2, hy2, hz2 = zero, zero, zero
                
            # the cluster velocity proxy of the galaxy
            dT2 = dT[j]

            # distance between galaxies
            dx = x1-x2
            dy = y1-y2
            dz = z1-z2
            dist = np.sqrt(dx**two + dy**two + dz**two)

            # index where this pair belongs
            if is_log_bin:
                ind = np.int64(np.floor(np.log(dist/fbin)/np.log(dbin)))
            else:
                ind = np.int64(np.floor((dist-fbin)/dbin))
                
            if (ind < len(bins)-1) and (ind >= 0):
                # norm of vector connecting galaxies
                if dist > zero:
                    hx, hy, hz = dx/dist, dy/dist, dz/dist
                else:
                    hx, hy, hz = zero, zero, zero

                # geometrical factor
                p12 = ((hx1+hx2)*hx + (hy1+hy2)*hy + (hz1+hz2)*hz)*half

                # record the sums for each pair
                pair_count[t, ind] += two #one
                weight_count[t, ind] += (dT1-dT2) * p12
                norm_weight_count[t, ind] += p12**two

    # compute counts per radial bin
    pair_count = pair_count.sum(axis=0)
    weight_count = weight_count.sum(axis=0)
    norm_weight_count = norm_weight_count.sum(axis=0)
    #if fbin == zero: pair_count[0] -= len(dT) # unnecessary because i < j
    
    # obtain pairwise momentum
    for i in range(len(bins)-1):
        if norm_weight_count[i] != zero:
            pairwise_velocity[i] = -weight_count[i]/norm_weight_count[i]

    return pair_count, pairwise_velocity

@numba.njit(parallel=True, fastmath=True)
def pairwise_velocity_sky(X, V_los, Lbox, bins, is_log_bin, dtype=np.float32, nthread=1):
    """
    The formula is:
    v_1,2 = 2 Sum_A,B (s_A-s_B) p_AB / (Sum_A,B p_AB^2)
    p_AB = \hat r . (\hat r_A + \hat r_B)
    \vec r = \vec r_A - \vec r_B
    s_A = \hat r_A v_A
    V_los is independently measured
    Does not use periodic boundary conditions
    """
    # TODO: check that the pairwise velocity thingy works
    
    numba.set_num_threads(nthread)

    N = len(X)
    Lbox = dtype(Lbox)
    zero = dtype(0.)
    one = dtype(1.)
    two = dtype(2.)

    # only works for linear
    if is_log_bin:
        dbin = dtype(bins[1]/bins[0])
    else:
        dbin = dtype(bins[1]-bins[0])
    fbin = dtype(bins[0])
    
    pair_count = np.zeros((nthread, len(bins)-1), dtype=dtype)
    weight_count = np.zeros((nthread, len(bins)-1), dtype=dtype)
    norm_weight_count = np.zeros((nthread, len(bins)-1), dtype=dtype)
    pairwise_velocity = np.zeros(len(bins)-1, dtype=dtype)
    
    for i in numba.prange(N):
        x1, y1, z1 = X[i][0], X[i][1], X[i][2]
        dist1 = np.sqrt(x1**two + y1**two + z1**two)

        if dist1 > zero:
            hx1, hy1, hz1 = x1/dist1, y1/dist1, z1/dist1
        else:
            hx1, hy1, hz1 = zero, zero, zero
        
        s1 = V_los[i]
        
        t = numba.np.ufunc.parallel._get_thread_id()
        
        for j in range(N):
            x2, y2, z2 = X[j][0], X[j][1], X[j][2]
            dist2 = np.sqrt(x2**two + y2**two + z2**two)

            if dist2 > zero:
                hx2, hy2, hz2 = x2/dist2, y2/dist2, z2/dist2
            else:
                hx2, hy2, hz2 = zero, zero, zero
                
            s2 = V_los[j]
            
            dx = x1-x2
            dy = y1-y2
            dz = z1-z2

            """
            if dx > Lbox/two:
                dx -= Lbox
            if dy > Lbox/two:
                dy -= Lbox
            if dz > Lbox/two:
                dz -= Lbox
            if dx <= -Lbox/two:
                dx += Lbox
            if dy <= -Lbox/two:
                dy += Lbox
            if dz <= -Lbox/two:
                dz += Lbox
            """
            dist2 = dx**two + dy**two + dz**two
            dist = np.sqrt(dist2)
            
            
            if is_log_bin:
                ind = np.int64(np.floor(np.log(dist/fbin)/np.log(dbin)))
            else:
                ind = np.int64(np.floor((dist-fbin)/dbin))
                
            if (ind < len(bins)-1) and (ind >= 0):
                if dist > zero:
                    hx, hy, hz = dx/dist, dy/dist, dz/dist
                else:
                    hx, hy, hz = zero, zero, zero

                p12 = (hx1+hx2)*hx + (hy1+hy2)*hy + (hz1+hz2)*hz

                pair_count[t, ind] += one
                weight_count[t, ind] += two * (s1-s2) * p12
                norm_weight_count[t, ind] += p12**two

    pair_count = pair_count.sum(axis=0)
    weight_count = weight_count.sum(axis=0)
    norm_weight_count = norm_weight_count.sum(axis=0)
    if fbin == zero: pair_count[0] -= len(V_los)
    
    for i in range(len(bins)-1):
        if norm_weight_count[i] != zero:
            pairwise_velocity[i] = weight_count[i]/norm_weight_count[i]
    

    return pair_count, pairwise_velocity


@numba.njit(parallel=True, fastmath=True)
def pairwise_velocity_box(X, V_los, Lbox, bins, is_log_bin, dtype=np.float32, nthread=1):
    """
    The formula is:
    v_1,2 = 2 Sum_A,B (s_A-s_B) p_AB / (Sum_A,B p_AB^2)
    p_AB = \hat r . (\hat r_A + \hat r_B)
    \vec r = \vec r_A - \vec r_B
    s_A = \hat r_A v_A
    
    Here we assume that the line-of-sight is along z and that we are in a
    box with periodic boundary conditions. Thus:
    \hat r_A = \hat r_B = 1
    Thus p_AB = 2 \hat r_z
    Here we are introducing V_los separately as an independently measured quantity
    Assumes periodic boundary conditions
    """
    # TODO: check that the pairwise velocity thingy works
    
    numba.set_num_threads(nthread)

    N = len(X)
    Lbox = dtype(Lbox)
    zero = dtype(0.)
    one = dtype(1.)
    two = dtype(2.)

    # only works for linear
    if is_log_bin:
        dbin = dtype(bins[1]/bins[0])
    else:
        dbin = dtype(bins[1]-bins[0])
    fbin = dtype(bins[0])
    
    pair_count = np.zeros((nthread, len(bins)-1), dtype=dtype)
    weight_count = np.zeros((nthread, len(bins)-1), dtype=dtype)
    norm_weight_count = np.zeros((nthread, len(bins)-1), dtype=dtype)
    pairwise_velocity = np.zeros(len(bins)-1, dtype=dtype)
    
    for i in numba.prange(N):
        x1, y1, z1 = X[i][0], X[i][1], X[i][2]
        s1 = V_los[i]

        t = numba.np.ufunc.parallel._get_thread_id()
        
        for j in range(N):
            x2, y2, z2 = X[j][0], X[j][1], X[j][2]
            s2 = V_los[j]
            
            dx = x1-x2
            dy = y1-y2
            dz = z1-z2

            if dx > Lbox/two:
                dx -= Lbox
            if dy > Lbox/two:
                dy -= Lbox
            if dz > Lbox/two:
                dz -= Lbox
            if dx <= -Lbox/two:
                dx += Lbox
            if dy <= -Lbox/two:
                dy += Lbox
            if dz <= -Lbox/two:
                dz += Lbox

            dist2 = dx**two + dy**two + dz**two
            dist = np.sqrt(dist2)

            if is_log_bin:
                ind = np.int64(np.floor(np.log(dist/fbin)/np.log(dbin)))
            else:
                ind = np.int64(np.floor((dist-fbin)/dbin))
                
            if (ind < len(bins)-1) and (ind >= 0):
                if dist > zero:
                    hat_dz = dz/dist
                else:
                    hat_dz = zero

                p12 = two * hat_dz
                pair_count[t, ind] += one
                weight_count[t, ind] += two * (s1-s2) * p12
                norm_weight_count[t, ind] += p12**two

    pair_count = pair_count.sum(axis=0)
    weight_count = weight_count.sum(axis=0)
    norm_weight_count = norm_weight_count.sum(axis=0)
    if fbin == zero: pair_count[0] -= len(V_los)
    
    for i in range(len(bins)-1):
        if norm_weight_count[i] != zero:
            pairwise_velocity[i] = weight_count[i]/norm_weight_count[i]
    
    return pair_count, pairwise_velocity


def pairwise_vel_symm(pos, v1d, bins, nthread, periodic=False, box=None, pos2=None, v1d2=None, isa='avx512f'):#'fallback'):#'avx'):
    """
    If isa="avx" or "avx512f" (fastest) is not implemented, would "fallback" to the DOUBLE implementation
    """
    
    # determine auto or cross correlation
    if v1d2 is not None:
        autocorr = 0
    else:
        autocorr = 1
    
    # combine position and velocity into weights
    rv = np.vstack((pos.T, v1d)).T
    if autocorr:
        rv2 = rv; pos2 = pos
    else:
        rv2 = np.vstack((pos2.T, v1d2)).T
    
    # compute numerator and denominator
    res = Corrfunc.theory.DD(autocorr, nthread, bins, *pos.T, weights1=rv.T, periodic=periodic, boxsize=box, X2=pos2[:, 0], Y2=pos2[:, 1], Z2=pos2[:, 2], weights2=rv2.T, verbose=False, isa=isa, weight_type='pairwise_vel_los')
    res_norm = Corrfunc.theory.DD(autocorr, nthread, bins, *pos.T, weights1=pos.T, periodic=periodic, boxsize=box, X2=pos2[:, 0], Y2=pos2[:, 1], Z2=pos2[:, 2], weights2=pos2.T, verbose=False, isa=isa, weight_type='pairwise_vel_los_norm')
    pairwise = -res['weightavg']/res_norm['weightavg']
    pairwise[res_norm['weightavg'] == 0.] = 0.
    
    # return pairwise estimator
    return pairwise

def pairwise_vel_asymm(pos, deltaT, bins, nthread, periodic=False, box=None, tau=None, pos2=None, bias2=None, isa='avx512f'):#'fallback'):#'avx'):
    """
    If isa="avx" or "avx512f" (fastest) is not implemented, would "fallback" to the DOUBLE implementation
    pairwise_vel_los_asymm requires 4 weights (3 positions, fourth is deltaT times tau prox for first and bias for second)
    """
    
    # must be cross correlation because asymmetric
    autocorr = 0
    if pos2 is None:
        pos2 = pos
    if tau is None:
        tau = np.ones(pos.shape[0]) 
    if bias2 is None:
        bias2 = np.ones(pos2.shape[0])
    
    # combine position and velocity into weights
    rv_num1 = np.vstack((pos.T, deltaT*tau)).T
    #rv_den1 = np.vstack((pos.T, tau)).T # not used
    rv_num2 = np.vstack((pos2.T, bias2)).T
    #rv_den2 = np.vstack((pos2.T, bias2)).T # not used
    pos2 = pos2.astype(np.float32)
    pos = pos.astype(np.float32)
    rv_num1 = rv_num1.astype(np.float32)
    rv_num2 = rv_num2.astype(np.float32)
    
    # compute numerator and denominator
    res = Corrfunc.theory.DD(autocorr, nthread, bins, *pos.T, weights1=rv_num1.T, periodic=periodic, boxsize=box, X2=pos2[:, 0], Y2=pos2[:, 1], Z2=pos2[:, 2], weights2=rv_num2.T, verbose=False, isa=isa, weight_type='pairwise_vel_los_asymm')
    res_norm = Corrfunc.theory.DD(autocorr, nthread, bins, *pos.T, weights1=pos.T, periodic=periodic, boxsize=box, X2=pos2[:, 0], Y2=pos2[:, 1], Z2=pos2[:, 2], weights2=pos2.T, verbose=False, isa=isa, weight_type='pairwise_vel_los_norm')
    #res_norm = Corrfunc.theory.DD(autocorr, nthread, bins, *pos.T, weights1=rv_den1.T, periodic=periodic, boxsize=box, X2=pos2[:, 0], Y2=pos2[:, 1], Z2=pos2[:, 2], weights2=rv_den2.T, verbose=False, isa=isa, weight_type='pairwise_vel_los_asymm_norm')
    pairwise = -res['weightavg']/res_norm['weightavg']
    pairwise[res_norm['weightavg'] == 0.] = 0.
    
    # return pairwise estimator
    return pairwise
