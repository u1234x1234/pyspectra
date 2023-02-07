import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

import pyspectra


def get_sparse_sym(size):
    x = sparse.random(size, 1000, density=0.01, format="csr", dtype=np.float32)
    return x.dot(x.T)


def get_dense_sym(size):
    x = np.random.normal(0, 20, size=(size, 1000)).astype(np.float32)
    return x.dot(x.T)


def test_scipy_eigsh():
    x = get_dense_sym(100)
    k = 16

    for which in ["LM", "SM", "LA", "SA", "BE"]:
        evalues, evectors = eigsh(x, k=k, which=which)
        indices = np.argsort(-evalues)
        evalues_gt = evalues[indices]
        evectors_gt = evectors.T[indices].T

        evalues, evectors = pyspectra.eigsh(x, k, backend="numpy", which=which)

        assert np.allclose(evalues_gt, evalues, rtol=0.1, atol=0.1)
        assert np.allclose(np.abs(evectors_gt), np.abs(evectors), rtol=0.1, atol=0.1)  # Up to sign
