import spectra_ext
from abc import ABC, abstractmethod
import cupy


class MatProdBackend(ABC):
    @abstractmethod
    def __init__(self, mat):
        pass

    @abstractmethod
    def matrix_vector_product(self, x, y):
        pass


class NumpyBackend(MatProdBackend):
    def __init__(self, mat):
        self._mat = mat

    def matrix_vector_product(self, x, y):
        y[:] = self._mat.dot(x)


class CupyBackend(MatProdBackend):
    def __init__(self, mat):
        self._mat = cupy.asarray(mat)

        self._x = cupy.zeros(mat.shape[0], dtype=mat.dtype)
        self._y = cupy.zeros(mat.shape[0], dtype=mat.dtype)

    def matrix_vector_product(self, x, y):
        self._x.set(x)
        cupy.dot(self._mat, self._x, self._y)
        y[:] = self._y.get()


class TorchBackend(MatProdBackend):
    pass


def eigs(x, n_top, backend="eigen"):
    if backend == "eigen":
        f = spectra_ext.eigs_sym_dense_float64
        args = ()
    elif backend == "numpy":
        f = spectra_ext.eigs_python_backend_float64
        args = (NumpyBackend,)
    elif backend == "cupy":
        f = spectra_ext.eigs_python_backend_float64
        args = (CupyBackend,)
    elif backend == "pytorch":
        f = spectra_ext.eigs_python_backend_float64
        args = (TorchBackend,)
    elif isinstance(backend, MatProdBackend):
        f = spectra_ext.eigs_python_backend_float64
        args = (backend,)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    ncv = n_top * 2
    return f(x, n_top, ncv, *args)
