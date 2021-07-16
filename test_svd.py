import numpy as np
from scipy.sparse.linalg import svds
from uxils.time import Timer

from pyspectra import partial_svd

x = np.random.normal(0, 20, size=(4000, 4000)).astype(np.float32)
n_top = 128

with Timer("arpack"):
    u, s, v = svds(x, n_top, solver="arpack")
print(u.shape, s.shape, v.shape)

with Timer("spectra"):
    u2, s2, v2 = partial_svd(x.T, n_top)
print(u2.shape, s2.shape, v2.shape)
