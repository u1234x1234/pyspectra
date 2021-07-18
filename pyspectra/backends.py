import importlib
from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse

# External will only be imported in case of actual using
cupy = None
cupyx = None  # cupy sparse
torch = None
tensorflow = None
jax = None
jnp = None  # import jax.numpy as jnp


def _load_module(name, g_name=None):
    try:
        m = importlib.import_module(name)
    except ModuleNotFoundError as e:
        raise ValueError(f"Please check the module '{name}' is installed.") from e

    if g_name is None:
        g_name = name
    globals()[g_name] = m


class MatProdBackend(ABC):
    @abstractmethod
    def __init__(self, mat):
        pass

    @abstractmethod
    def matrix_vector_product(self, x, y):
        pass


class DenseNumpyBackend(MatProdBackend):
    def __init__(self, mat):
        self._mat = mat

    def matrix_vector_product(self, x, y):
        y[:] = self._mat.dot(x)


class DenseCupyBackend(MatProdBackend):
    def __init__(self, mat):
        self._mat = cupy.asarray(mat)

        self._x = cupy.zeros(mat.shape[0], dtype=mat.dtype)
        self._y = cupy.zeros(mat.shape[0], dtype=mat.dtype)

    def matrix_vector_product(self, x, y):
        self._x.set(x)
        cupy.dot(self._mat, self._x, self._y)
        y[:] = self._y.get()


class DenseTorchBackend(MatProdBackend):
    def __init__(self, mat):
        self._mat = torch.from_numpy(mat).cuda()

    def matrix_vector_product(self, x, y):
        x = torch.from_numpy(x).cuda()
        yt = torch.from_numpy(y).cuda()
        torch.mv(self._mat, x, out=yt)
        y[:] = yt.cpu().numpy()


class DenseTensorflowBackend(MatProdBackend):
    def __init__(self, mat):
        self._mat = tensorflow.constant(mat)

    def matrix_vector_product(self, x, y):
        r = tensorflow.linalg.matvec(self._mat, x)
        y[:] = r.numpy()


class DenseJaxBackend(MatProdBackend):
    def __init__(self, mat):
        self._mat = jax.device_put(mat)

    def matrix_vector_product(self, x, y):
        y[:] = jax.numpy.dot(self._mat, x)


class SparseScipyBackend(MatProdBackend):
    def __init__(self, mat):
        self._mat = sparse.csr_matrix(mat)

    def matrix_vector_product(self, x, y):
        y[:] = self._mat.dot(x)


def _jax_dot_product(rows, cols, data, b):
    return jnp.zeros_like(b).at[rows].add(data * b[cols])


class SparseJaxBackend(MatProdBackend):
    def __init__(self, mat):
        m = sparse.coo_matrix(mat)
        self._row = jax.device_put(m.row)
        self._col = jax.device_put(m.col)
        self._data = jax.device_put(m.data)
        self._dp = jax.jit(_jax_dot_product)

    def matrix_vector_product(self, x, y):
        y[:] = self._dp(self._row, self._col, self._data, x)


class SparseCupyBackend(MatProdBackend):
    def __init__(self, mat):
        self._mat = cupyx.scipy.sparse.csr_matrix(mat)

    def matrix_vector_product(self, x, y):
        x = cupyx.scipy.sparse.csr_matrix(sparse.csr_matrix(x[:, np.newaxis]))
        y[:] = self._mat.dot(x).todense().get().ravel()


DENSE_BACKENDS = {
    "numpy": DenseNumpyBackend,
    "cupy": DenseCupyBackend,
    "torch": DenseTorchBackend,
    "tensorflow": DenseTensorflowBackend,
    "jax": DenseJaxBackend,
}

SPARSE_BACKENDS = {
    "scipy": SparseScipyBackend,
    "cupy": SparseCupyBackend,
    "jax": SparseJaxBackend,
}


def get_backend(backend, matrix_type) -> type:

    if isinstance(backend, str):
        if matrix_type == "sparse":
            backend_cls = SPARSE_BACKENDS.get(backend)
            if backend_cls is None:
                raise ValueError(
                    f"No such backend: '{backend}' for sparse matrices. Available options: {list(SPARSE_BACKENDS.keys())}"
                )

        elif matrix_type == "dense":
            backend_cls = DENSE_BACKENDS.get(backend)
            if backend_cls is None:
                raise ValueError(
                    f"No such backend: '{backend}' for dense matrices. Available options: {list(DENSE_BACKENDS.keys())}"
                )

        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}")

        if backend in ("jax", "torch", "tensorflow", "cupy"):
            _load_module(backend)
            if backend == "cupy" and matrix_type == "sparse":
                _load_module("cupyx")
            if backend == "jax" and matrix_type == "sparse":
                _load_module("jax")
                _load_module("jax.numpy", "jnp")

    else:
        assert issubclass(backend, MatProdBackend)
        backend_cls = backend

    return backend_cls
