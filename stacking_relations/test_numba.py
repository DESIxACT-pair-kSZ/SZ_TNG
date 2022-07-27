from numba_2pcf.cf import numba_pairwise_vel, numba_2pcf

import numpy as np

# define bins in Mpc
rbins = np.linspace(0., 30., 11) # Mpc/h
rbinc = (rbins[1:]+rbins[:-1])*.5 # Mpc/h
nthread = 1 #os.cpu_count()//4

pos = np.random.rand(20000, 3)*205.
delta_Ts = np.ones(pos.shape[0])

print(pos[:, 0].min(), pos[:, 1].min(), pos[:, 2].min(), pos[:, 0].max(), pos[:, 1].max(), pos[:, 2].max(), delta_Ts.min(), delta_Ts.max(), delta_Ts.mean())

table = numba_pairwise_vel(pos, delta_Ts, box=None, Rmax=np.max(rbins), nbin=len(rbins)-1, corrfunc=False, nthread=nthread, periodic=False)
#table = numba_2pcf(pos, box=205., Rmax=np.max(rbins), nbin=len(rbins)-1, corrfunc=False, nthread=nthread)
DD = table['npairs']
print(DD)
