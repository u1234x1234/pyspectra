import numpy as np
import spectra_ext
from scipy import sparse

from .backends import get_backend


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


def eigsh(x, k, ncv=None, maxiter=1000, backend="eigen"):
    "Find k eigenvalues and eigenvectors of the real symmetric square matrix"

    dtype_name, matrix_type_name = _detect_matrix_type(x)
    backend_class = get_backend(backend, matrix_type_name)

    func_name = f"eigs_python_backend_{dtype_name}"

    if backend == "eigen":
        func_name = f"sym_eigs_{matrix_type_name}_eigen_{dtype_name}"
        args = ()
    else:
        func_name = f"sym_eigs_{matrix_type_name}_pybackend_{dtype_name}"
        args = (backend_class,)

    f = spectra_ext.__dict__[func_name]
    ncv = k * 2 if ncv is None else ncv
    evalues, evectors, status = f(x, k, ncv, maxiter, *args)

    if status == 2:
        raise ValueError("NotConverging")
    elif status == 3:
        raise ValueError("NumericalIssue")

    return evalues, evectors


def partial_svd(x, k, ncv=None):
    dtype_name, matrix_type_name = _detect_matrix_type(x)

    if ncv is None:
        ncv = k * 2

    func_name = f"partial_svd_{dtype_name}"
    r = spectra_ext.__dict__[func_name](x, k, ncv)
    return r
