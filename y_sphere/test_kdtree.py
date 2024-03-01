import itertools

import numpy as np
from scipy import spatial

def concat_to_arr(lists, dtype=np.int64):
    '''Concatenate an iterable of lists to a flat Numpy array.
    Returns the concatenated array and the index where each list starts.
    '''
    starts = np.empty(len(lists) + 1, dtype=np.int64)
    starts[0] = 0
    starts[1:] = np.cumsum(np.fromiter((len(l) for l in lists), count=len(lists), dtype=np.int64))
    N = starts[-1]
    res = np.fromiter(itertools.chain.from_iterable(lists), count=N, dtype=dtype)
    return res, starts

Lbox_hkpc = 500000. # ckpc/h
C = np.random.uniform(0., Lbox_hkpc, (1000, 3))
R = np.random.uniform(1000., 2000., 1000)
dV = np.random.uniform(1000., 2000., 1000)

tree = spatial.cKDTree(C, boxsize=Lbox_hkpc)

rbins = np.logspace(-2, 2, 21)*1000.
r = R*100.
#rbins[None, :]*R[:, None]
print(r.shape)


inds = tree.query_ball_point(x=C, r=r)#, workers=16)
#inner_arr, inner_starts = concat_to_arr(inds)
print(len(inds))
lens = tree.query_ball_point(x=C, r=r, return_length=True)
print(np.max(lens))

starts = np.zeros(len(lens))
starts[1:] = np.cumsum(lens)[:-1]
inds = np.hstack(inds)
#inds = np.asarray(inds).flatten()
print(inds.shape)
print(starts.shape)
quit()

# add zeros to final entry of array

inds_arr = np.zeros((C.shape[0], np.max(lens)), dtype=np.int)-1
for i in range(C.shape[0]):
    inds_arr[i, :lens[i]] = np.asarray(inds[i]).flatten()
print(inds_arr)

V = np.sum(dV[inds_arr], axis=1)
print("should be 1000 and not max len", V.shape)
#print(inds, lens)
#np.asarray

#for i in range(C.shape[0]):
#    ind = np.asarray(inds[i]).flatten()
#    print(len(ind))

quit()
print(inds.shape)
print(inds[0])
print(np.asarray(inds[0]).flatten())
print(inds.shape)
