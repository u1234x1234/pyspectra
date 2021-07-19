import pickle
from collections import defaultdict

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from uxils.time.benchmark import benchmark_func
from itertools import product
import pyspectra
from pyspectra import list_dense_backends, list_sparse_backends


def get_sparse_sym(size):
    x = sparse.random(size, 1000, density=0.01, format="csr", dtype=np.float32)
    return x.dot(x.T)


def get_dense_sym(size):
    x = np.random.normal(0, 20, size=(size, 1000)).astype(np.float32)
    return x.dot(x.T)


for mat_type, n_values in product(["dense", "sparse"], [16]):
    g_res = defaultdict(dict)

    for mat_size in [100, 1_000, 2_000, 10_000, 20_000]:

        if mat_type == "dense":
            x = get_dense_sym(mat_size)
        else:
            x = get_sparse_sym(mat_size)

        def scipy_eigsh():
            return eigsh(x, k=n_values)

        (evalues, evectors), measurements = benchmark_func(scipy_eigsh, 1, sandbox=0)

        indices = np.argsort(-evalues)
        evalues_gt = evalues[indices]
        evectors_gt = evectors.T[indices].T
        g_res["scipy.eigsh, ARPACK"][mat_size] = measurements

        backends = (
            list_dense_backends() if mat_type == "dense" else list_sparse_backends()
        )
        for backend in backends:
            print(mat_type, n_values, mat_size, backend)

            def pyspectra_eigsh():
                return pyspectra.eigsh(x, n_values, backend=backend)

            (evalues, evectors), measurements = benchmark_func(
                pyspectra_eigsh, 1, sandbox=1
            )

            assert np.allclose(evalues_gt, evalues, rtol=0.1, atol=0.1)
            assert np.allclose(
                np.abs(evectors_gt), np.abs(evectors), rtol=0.1, atol=0.1
            )  # Up to sign

            if backend == "eigen":
                name = "Spectra default [Eigen]"
            else:
                name = f"pyspectra with {backend} backend"

            g_res[name][mat_size] = measurements

    with open(f"measurements_{mat_type}_{n_values}.pkl", "wb") as out_file:
        pickle.dump(g_res, out_file)
