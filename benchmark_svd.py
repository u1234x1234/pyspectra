import pyspectra
import numpy as np
from uxils.time import Timer
from scipy.sparse.linalg import svds

N_COMP = 16

for data_shape in [(4096, 20_000), (20_000, 4096)]:

    X = np.random.normal(0, 20, size=data_shape).astype(np.float32)

    # with Timer("scipy"):
    #     U, s, Vt = svds(X, k=N_COMP)
    # indices = np.argsort(-s)
    # U = U.T[indices].T
    # V = Vt[indices].T
    # s = s[indices]

    for backend in ["jax", "torch"]:
        U2, s2, V2 = pyspectra.partial_svd(X, N_COMP, backend=backend)

        with Timer(backend):
            U2, s2, V2 = pyspectra.partial_svd(X, N_COMP, backend=backend)

        # assert np.allclose(s, s2, rtol=0.1, atol=0.1)
        # assert np.allclose(np.abs(U), np.abs(U2), rtol=0.2, atol=0.2)
        # assert np.allclose(np.abs(V), np.abs(V2), rtol=0.2, atol=0.2)
