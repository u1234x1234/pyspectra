import pickle

import numpy as np
from scipy.sparse.linalg import svds
from uxils.time import Timer

import pyspectra

N_COMP = 32


def torch_warmup():
    import torch

    torch.zeros(10_000).cuda()


torch_warmup()
for data_shape in [(10_000, 4000)]:
    results = {}

    X = np.random.normal(0, 20, size=data_shape).astype(np.float32)

    with Timer("scipy") as t:
        U, s, Vt = svds(X, k=N_COMP)

    results["scipy.svds[ARPACK]"] = t.elapsed

    indices = np.argsort(-s)
    U = U.T[indices].T
    V = Vt[indices].T
    s = s[indices]

    for backend in ["numpy", "torch"]:

        with Timer(backend) as t:
            U2, s2, V2 = pyspectra.truncated_svd(X, N_COMP, backend=backend)

        results[f"pyspectra [{backend}]"] = t.elapsed
        assert np.allclose(s, s2, rtol=0.1, atol=0.1)
        assert np.allclose(np.abs(U), np.abs(U2), rtol=0.2, atol=0.2)
        assert np.allclose(np.abs(V), np.abs(V2), rtol=0.2, atol=0.2)

    with open(f"measurements_svd_{data_shape}_{N_COMP}.pkl", "wb") as out_file:
        pickle.dump(results, out_file)
