# pyspectra - Unofficial python interface to Spectra library; GPU accelerated eigenvalue problems solving

One of the features of Spectra library is ability to customize matrix operations - to solve eigenvalue problems you only need to specify matrix-vector product operation.
By default Spectra uses Eigen library for computations.
`pyspectra` allows you to redefine matrix-vector operation using python code. For example you can utilize external libraries with GPU support to make all matrix computations which leads to large speedups.

Please open an issue you want to use some solver which is not supported

Example of dense matrix-vector product with `numpy`:
```python
class DenseNumpyBackend:
    def __init__(self, mat):
        self._mat = mat

    def matrix_vector_product(self, x, out):
        out[:] = self._mat.dot(x)
```

If you have a GPU you can required matrix operations to Pytorch:
```python
class DenseTorchBackend:
    def __init__(self, mat):
        self._mat = torch.from_numpy(mat).cuda()

    def matrix_vector_product(self, x, out):
        x = torch.from_numpy(x).cuda()
        yt = torch.from_numpy(y).cuda()
        torch.mv(self._mat, x, out=yt)
        out[:] = yt.cpu().numpy()
```

## Implemented Spectra solvers

* [SymEigsSolver](https://spectralib.org/doc/classSpectra_1_1SymEigsSolver.html) - real symmetric dense/sparse
* [PartialSVDSolver](https://github.com/yixuan/spectra/blob/master/include/Spectra/contrib/PartialSVDSolver.h) - partial SVD without explicit A.t * A construction

## Backends

### Dense matrix-vector multiplication

* Numpy
* [PyTorch](https://pytorch.org/)
* cupy
* Tensorflow
* Jax

### Sparse matrix-vector multiplication

* Scipy
* Cupy
* Jax

## Performance Note

Actually it is not the fastest library for basic usage (e.g real symmetric eigen solver) as it introduces overheads: cpu/gpu memory copies and c++/python interoperability.

For example for cupy backend the sequence of calls will be similar to:
1. your python code
2. pyspectra python code
3. pyspectra c++ code
4. spectra c++ code
5. user-defined matrix ops in python
6. cupy python code which calls cublas
7. actual cublas code (executed on GPU)
