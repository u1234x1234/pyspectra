import pyspectra
import numpy as np
from uxils.time import Timer
from scipy.sparse.linalg import eigsh
from spectra_ext import eigs2
import cupy

with Timer("init"):
    cupy.array(np.zeros(10000))
n = 0

def func(mat, x, y):
    y[:] = mat.dot(x)


def func(mat, x, y):
    # mat = cupy.array(mat)
    with Timer("cupy"):
        global n
        n += 1
        # x1 = cupy.array(x)

    y[:] = mat.dot(x)


# r = pyspectra.eigs(func, np.array([1, 2, 3], dtype=np.float32))
# print(r)
# qwe

x = np.random.uniform(-20, 20, size=(2000, 2000))
x = x.dot(x.T)
print(x.shape)


with Timer("scipy"):
    r2 = eigsh(x, k=3)
print(r2[0][:3])

with Timer("spectra"):
    r = eigs2(x.T, func)

print(r[:3])
print(n)

# with Timer("np"):
#     vec, val = np.linalg.eig(x)
# print(vec[:5])
