import pyspectra
import numpy as np
from uxils.time import Timer
from scipy.sparse.linalg import eigsh

x = np.random.uniform(-20, 20, size=(5000, 100))
x = x.dot(x.T)
print(x.shape)

with Timer("scipy"):
    r2 = eigsh(x, k=3)

print(r2[0][:3])

with Timer("spectra"):
    r = pyspectra.eigs(x.T)
print(r)

with Timer("np"):
    vec, val = np.linalg.eig(x)
print(vec[:5])
