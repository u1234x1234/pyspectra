# Unofficial python interface to Spectra library; GPU accelerated eigenvalue / Truncated (partial) SVD problems solving

[Spectra](https://github.com/yixuan/spectra) is a C++ library for large scale eigenvalue problems.

<img src="https://i.imgur.com/vAbxDdq.png" width="700">
<img src="https://i.imgur.com/YxmIHcT.png" width="700">

By default Spectra uses [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) library for computations, but exploiting its design it is possible to outsource linear algebra computations to any external library.
`pyspectra` allows you to redefine matrix-vector operation using python code. For example you can utilize libraries with GPU support (such as PyTorch, TensorFlow, CuPy) which leads to large speedups.

## Installation

```bash
pip install git+https://github.com/u1234x1234/pyspectra.git@0.0.1
```

You need to have [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) library installed. (`fatal error: Eigen/Core: No such file or directory`)

Installation using `apt`:
```
sudo apt install libeigen3-dev
```

## Usage

```python
import pyspectra
import numpy as np

X = np.random.uniform(size=(10_000, 1000))

U, s, V = pyspectra.truncated_svd(X, 20)  # similar to scipy.sparse.linalg.svds; Eigen
U, s, V = pyspectra.truncated_svd(X, 20, backend="torch")  # GPU acceleration with PyTorch

eigenvalues, eigenvectors = pyspectra.eigsh(X.T.dot(X), 20)  # Symmetric eigenvalue problem, scipy.sparse.linalg.eigsh
```

## Implemented Spectra solvers

* [SymEigsSolver](https://spectralib.org/doc/classSpectra_1_1SymEigsSolver.html) - real symmetric; dense/sparse; float32/float64
* [PartialSVDSolver](https://github.com/yixuan/spectra/blob/master/include/Spectra/contrib/PartialSVDSolver.h) - partial SVD without explicit A.t * A construction; float32/float64

Please open an issue you want to use some solver which is not supported yet.

## Backends

### Dense matrix-vector

* Numpy
* [PyTorch](https://pytorch.org/)
* cupy
* Tensorflow
* Jax

### Sparse matrix-vector

* Scipy
* Cupy
* Jax

### Dense SVD

* numpy
* PyTorch
* Jax

## Implementing your backend

Example of dense matrix-vector product with `numpy`:
```python
class DenseNumpyBackend:
    def __init__(self, mat):
        self._mat = mat

    def perform_op(self, x, out):
        out[:] = self._mat.dot(x)
```

GPU accelerated matrix-vector product with `PyTorch`
```python
import torch

class DenseTorchBackend:
    def __init__(self, mat):
        self._mat = torch.from_numpy(mat).cuda()

    def perform_op(self, x, out):
        x = torch.from_numpy(x).cuda()
        yt = torch.from_numpy(y).cuda()
        torch.mv(self._mat, x, out=yt)
        out[:] = yt.cpu().numpy()
```

## Performance Note

Actually it is not the fastest library for basic usage (e.g real symmetric eigen solver) as it introduces overheads: cpu/gpu memory copies and c++/python interoperability.

For example for cupy backend the sequence of calls will be similar to:
1. your python code
2. pyspectra python code
3. pyspectra c++ code
4. spectra c++ code
5. user-defined matrix ops in python
6. cupy python code
7. actual number crunching with cublas
