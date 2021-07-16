import importlib
from abc import ABC, abstractmethod

import numpy as np
import spectra_ext

cupy = None
torch = None
tensorflow = None
jax = None


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
    def __init__(self, mat):
        self._mat = torch.from_numpy(mat).cuda()

    def matrix_vector_product(self, x, y):
        x = torch.from_numpy(x).cuda()
        yt = torch.from_numpy(y).cuda()
        torch.mv(self._mat, x, out=yt)
        y[:] = yt.cpu().numpy()


class TensorflowBackend(MatProdBackend):
    def __init__(self, mat):
        self._mat = tensorflow.constant(mat)

    def matrix_vector_product(self, x, y):
        r = tensorflow.linalg.matvec(self._mat, x)
        y[:] = r.numpy()


class JaxBackend(MatProdBackend):
    def __init__(self, mat):
        self._mat = jax.device_put(mat)

    def matrix_vector_product(self, x, y):
        y[:] = jax.numpy.dot(self._mat, x)


def check_dtype(x):
    if x.dtype == np.float32:
        suffix = "float32"
    elif x.dtype == np.float64:
        suffix = "float64"
    else:
        raise ValueError(f"Only float32/64 dtypes are supported now. Passed: {x.dtype}")

    return suffix


def _load_module(name):
    try:
        m = importlib.import_module(name)
    except ModuleNotFoundError as e:
        raise ValueError(f"Please check the module '{name}' is installed.") from e

    globals()[name] = m


def eigsh(x, n_top, maxiter=1000, backend="eigen"):
    "Find k eigenvalues and eigenvectors of the real symmetric square matrix"

    suffix = check_dtype(x)

    func_name = f"eigs_python_backend_{suffix}"
    if backend == "eigen":
        func_name = f"eigs_sym_dense_{suffix}"
        args = ()
    elif backend == "numpy":
        args = (NumpyBackend,)
    elif backend == "cupy":
        _load_module("cupy")
        args = (CupyBackend,)
    elif backend == "pytorch":
        _load_module("torch")
        args = (TorchBackend,)
    elif backend in ("tf", "tensorflow"):
        _load_module("tensorflow")
        args = (TensorflowBackend,)
    elif backend == "jax":
        _load_module("jax")
        args = (JaxBackend,)
    elif isinstance(backend, MatProdBackend):
        args = (backend,)
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Available options: eigen, numpy, pytorch, cupy"
        )

    f = spectra_ext.__dict__[func_name]
    ncv = n_top * 2
    evalues, evectors, status = f(x, n_top, ncv, maxiter, *args)

    if status == 2:
        raise ValueError("NotConverging")
    elif status == 3:
        raise ValueError("NumericalIssue")

    return evalues, evectors


def partial_svd(x, k, ncv=None):
    suffix = check_dtype(x)

    if ncv is None:
        ncv = k * 2

    func_name = f"partial_svd_{suffix}"
    r = spectra_ext.__dict__[func_name](x, k, ncv)
    return r
