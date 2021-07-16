import cupy
import numpy as np
from scipy import sparse
from scipy.sparse.csr import csr_matrix
import spectra_ext
from scipy.sparse.linalg import eigsh
from uxils.sparse import density
from uxils.time import Timer
import time
import numpy as np
# import torch
from uxils.pprint_ext import print_table
import pyspectra

# import jax.numpy as jnp
# x = np.zeros((10000, 5000))
# y = np.zeros((5000,))
# x = jnp.asarray(x)
# r = jnp.dot(x, y)
# print(np.array(r))
# qwe


# import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
    # tf.config.experimental.set_memory_growth(gpu, True)
# r = tf.linalg.matvec(x, y)

# with Timer():
#     r = tf.linalg.matvec(x, y)

# print(r.numpy().shape)
# cupy.zeros(1000*1000)
# torch.zeros(1000 * 1000).cuda()
# torch.set_grad_enabled(False)

# x = np.random.normal(0, 20, size=(100, 1000)).astype(np.float32)
# x = x.dot(x.T)
x = sparse.random(10000, 100, density=0.05, format="csr", dtype=np.float32)
x = x.dot(x.T)
print(x.shape)

# import cupyx
# d = cupyx.scipy.sparse.csr_matrix(x)
# y = np.zeros((10_000, 1), dtype=np.float32)
# y = cupyx.scipy.sparse.csr_matrix(csr_matrix(y))
# r = d.dot(y).todense().get()
# print(type(r))
# qwe

n_values = 64
maxiter = 1000
n_repetitions = 3

results = []

st = time.time()
evalues, evectors = eigsh(x, k=n_values)
results.append({"backend": "scipy ARPACK", "time": time.time() - st})
indices = np.argsort(-evalues)
evalues_scipy = evalues[indices]
evectors_scipy = evectors.T[indices].T

for backend in [
    "scipy",
    # "numpy",
    # "eigen",
    # "cupy",
    # "pytorch",
    # "tensorflow"
    "jax",
]:
    time_estimations = []
    for _ in range(n_repetitions):
        st = time.time()
        evalues, evectors = pyspectra.eigsh(x.T, n_values, backend=backend, maxiter=maxiter)
        st = time.time() - st
        time_estimations.append(st)

    results.append({"backend": backend, "time": np.median(time_estimations), "time std": np.std(time_estimations)})
    print_table(results)

    assert np.allclose(evalues_scipy, evalues, rtol=0.1, atol=0.1)
    assert np.allclose(np.abs(evectors_scipy), np.abs(evectors), rtol=0.1, atol=0.1)  # Up to sign
