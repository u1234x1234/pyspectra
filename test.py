from importlib import import_module
import time
import weld


import cupy
import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from uxils.pprint_ext import print_table

import pyspectra

# cupy.zeros(1000*1000)
torch.zeros(1000 * 1000).cuda()
torch.set_grad_enabled(False)

x = np.random.normal(0, 20, size=(5000, 2000)).astype(np.float32)
x = x.dot(x.T)
print(x.shape)
n_values = 32
maxiter = 1000
n_repetitions = 10

results = []

st = time.time()
evalues, evectors = eigsh(x, k=n_values)
results.append({"backend": "scipy ARPACK", "time": time.time() - st})
indices = np.argsort(-evalues)
evalues_scipy = evalues[indices]
evectors_scipy = evectors.T[indices].T

for backend in [
    # "numpy",
    # "eigen",
    # "cupy",
    "pytorch",
]:
    time_estimations = []
    for _ in range(n_repetitions):
        st = time.time()
        evalues, evectors = pyspectra.eigsh(x.T, n_values, backend=backend, maxiter=maxiter)
        st = time.time() - st
        time_estimations.append(st)

    results.append({"backend": backend, "time": np.mean(time_estimations), "time std": np.std(time_estimations)})
    print_table(results)

    assert np.allclose(evalues_scipy, evalues, rtol=0.1, atol=0.1)
    assert np.allclose(np.abs(evectors_scipy), np.abs(evectors), rtol=0.1, atol=0.1)  # Up to sign
