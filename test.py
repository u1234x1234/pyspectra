import numpy as np
from uxils.time import Timer
import pyspectra
import cupy

cupy.zeros(1000*1000)

x = np.random.normal(0, 20, size=(10_000, 1000))
x = x.dot(x.T)
print(x.shape)
n_values = 16


with Timer("scipy ARPACK"):
    from scipy.sparse.linalg import eigsh
    evalues, evectors = eigsh(x, k=n_values)
    indices = np.argsort(-evalues)
    evalues_scipy = evalues[indices]
    evectors_scipy = evectors.T[indices].T


results = []
for backend in ["numpy", "eigen", "cupy"]:
    with Timer(backend) as timer:
        evalues, evectors = pyspectra.eigs(x.T, n_values, backend=backend)

    assert np.allclose(evalues_scipy, evalues)
    assert np.allclose(np.abs(evectors_scipy), np.abs(evectors))  # Up to sign

    results.append({"backend": backend, "time": timer.elapsed})

    from uxils.pprint_ext import print_table
    print_table(results)
