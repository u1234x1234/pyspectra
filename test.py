import pickle
from collections import defaultdict

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from uxils.time.benchmark import benchmark_func

import pyspectra


def get_sparse_sym(size):
    x = sparse.random(size, 1000, density=0.01, format="csr", dtype=np.float32)
    return x.dot(x.T)


def get_dense_sym(size):
    x = np.random.normal(0, 20, size=(size, 500)).astype(np.float32)
    return x.dot(x.T)


n_values = 64
maxiter = 1000
n_repetitions = 3
g_res = defaultdict(dict)

for mat_size in [100, 1000, 2000, 10_000, 30_000]:
    x = get_dense_sym(mat_size)

    def scipy_eigsh():
        return eigsh(x, k=n_values)

    (evalues, evectors), measurements = benchmark_func(scipy_eigsh, 1, sandbox=0)

    indices = np.argsort(-evalues)
    evalues_gt = evalues[indices]
    evectors_gt = evectors.T[indices].T
    g_res["scipy.eigsh[arpack]"][mat_size] = measurements

    for backend in [
        # "scipy",
        "numpy",
        "eigen",
        "cupy",
        "torch",
        "tensorflow",
        "jax",
    ]:

        def pyspectra_eigsh():
            return pyspectra.eigsh(x.T, n_values, backend=backend, maxiter=maxiter)

        (evalues, evectors), measurements = benchmark_func(
            pyspectra_eigsh, 1, sandbox=1
        )

        print(mat_size, backend)
        assert np.allclose(evalues_gt, evalues, rtol=0.1, atol=0.1)
        assert np.allclose(
            np.abs(evectors_gt), np.abs(evectors), rtol=0.1, atol=0.1
        )  # Up to sign

        if backend == "eigen":
            name = "Spectra default [Eigen]"
        else:
            name = f"pyspectra with {backend} backend"

        g_res[name][mat_size] = measurements


with open("measurements2.pkl", "wb") as out_file:
    pickle.dump(g_res, out_file)
