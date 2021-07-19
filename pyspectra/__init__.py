import numpy as np
import spectra_ext
from scipy import sparse

from .backends import (
    DENSE_BACKENDS,
    SPARSE_BACKENDS,
    SVD_DENSE_BACKENDS,
    SVD_SPARSE_BACKENDS,
    get_backend,
    get_svd_backend,
)


def _detect_matrix_type(x):
    """
    1. float32 / float64
    2. dense / spare
    """

    if x.dtype == np.float32:
        dtype_name = "float32"
    elif x.dtype == np.float64:
        dtype_name = "float64"
    else:
        raise ValueError(f"Only float32/64 dtypes are supported now. Passed: {x.dtype}")

    if isinstance(x, np.ndarray):
        matrix_type_name = "dense"
    elif isinstance(x, sparse.spmatrix):
        matrix_type_name = "sparse"
    else:
        raise ValueError(
            f"Unknown matrix type: {type(x)}.\
            Available options: np.ndarray for dense, scipy.sparse.csr[csc]_matrix for sparse"
        )

    return dtype_name, matrix_type_name


def eigsh(x, k, backend="eigen", maxiter=1000, ncv=None):
    "Find k eigenvalues and eigenvectors of the real symmetric square matrix"
    assert x.shape[0] == x.shape[1], "Square matrix expected"

    dtype_name, matrix_type_name = _detect_matrix_type(x)

    func_name = f"eigs_python_backend_{dtype_name}"

    if backend == "eigen":
        func_name = f"sym_eigs_{matrix_type_name}_eigen_{dtype_name}"
        args = ()
    else:
        func_name = f"sym_eigs_{matrix_type_name}_pybackend_{dtype_name}"
        args = (get_backend(backend, matrix_type_name),)

    f = spectra_ext.__dict__[func_name]
    ncv = min(k * 2, len(x)) if ncv is None else ncv
    evalues, evectors, status = f(x, k, ncv, maxiter, *args)

    if status == 2:
        raise ValueError("NotConverging")
    elif status == 3:
        raise ValueError("NumericalIssue")

    return evalues, evectors


def truncated_svd(x, k, ncv=None, backend="eigen"):
    dtype_name, matrix_type_name = _detect_matrix_type(x)

    ncv = min(k * 2, len(x)) if ncv is None else ncv

    if backend == "eigen":
        func_name = f"partial_svd_{dtype_name}"
        args = ()
    else:
        func_name = f"partial_svd_pybackend_{dtype_name}"
        args = (get_svd_backend(backend, matrix_type_name),)

    f = spectra_ext.__dict__[func_name]
    r = f(x, k, ncv, *args)
    return r


def list_dense_backends():
    return ["eigen"] + list(DENSE_BACKENDS.keys())


def list_sparse_backends():
    return ["eigen"] + list(SPARSE_BACKENDS.keys())


def list_svd_dense_backends():
    return ["eigen"] + list(SVD_DENSE_BACKENDS.keys())


def list_svd_sparse_backends():
    return ["eigen"] + list(SVD_SPARSE_BACKENDS.keys())
