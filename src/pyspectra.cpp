#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include <iostream>
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/contrib/PartialSVDSolver.h>

using namespace std;
using namespace Spectra;

namespace py = pybind11;

// Aliases
template <typename T>
using ndarray = pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>;
template <typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
template <typename T>
using EigenVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using MapConstVec = Eigen::Map<const EigenVector<T>>;
template <typename T>
using MapVec = Eigen::Map<EigenVector<T>>;

template <typename Scalar_, int Uplo = Eigen::Lower>
class PythonDenseSymMatProd
{
public:
    using Scalar = Scalar_;

private:
    py::buffer_info info;
    py::object backend;

public:
    PythonDenseSymMatProd(ndarray<Scalar> array, py::object backend)
    {
        this->backend = backend(array);
        info = array.request();
    }

    Eigen::Index rows() const { return info.shape[0]; }
    Eigen::Index cols() const { return info.shape[1]; }

    void perform_op(const Scalar *x_in, Scalar *y_out) const
    {
        MapConstVec<Scalar> x(x_in, info.shape[1]);
        MapVec<Scalar> y(y_out, info.shape[0]);
        this->backend.attr("matrix_vector_product")(x, y);
    }
};

template <typename Scalar_, typename InMatrix, int Uplo = Eigen::Lower, int Flags = Eigen::ColMajor, typename StorageIndex = int>
class PythonSparseSymMatProd
{
public:
    using Scalar = Scalar_;
private:

    InMatrix m_mat;
    py::object backend;

public:
    PythonSparseSymMatProd(const InMatrix& mat, py::object backend) :
        m_mat(mat)
    {
        this->backend = backend(mat);
    }

    Eigen::Index rows() const { return m_mat.rows(); }
    Eigen::Index cols() const { return m_mat.cols(); }

    void perform_op(const Scalar *x_in, Scalar *y_out) const
    {
        MapConstVec<Scalar> x(x_in, m_mat.cols());
        MapVec<Scalar> y(y_out, m_mat.rows());
        this->backend.attr("matrix_vector_product")(x, y);
    }
};

template <typename num_t, typename Matrix>
Matrix ndarrayToMatrixX(pybind11::array_t<num_t, pybind11::array::c_style | pybind11::array::forcecast> b)
{
    typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> Strides;
    constexpr bool rowMajor = Matrix::Flags & Eigen::RowMajorBit;

    py::buffer_info info = b.request();

    if (info.format != py::format_descriptor<num_t>::format())
        throw std::runtime_error("Incompatible format: expected a double array!");
    if (info.ndim != 2)
        throw std::runtime_error("Incompatible buffer dimension!");

    auto strides = Strides(
        info.strides[rowMajor ? 0 : 1] / (py::ssize_t)sizeof(num_t),
        info.strides[rowMajor ? 1 : 0] / (py::ssize_t)sizeof(num_t));

    auto map = Eigen::Map<Matrix, 0, Strides>(
        static_cast<num_t *>(info.ptr), info.shape[0], info.shape[1], strides);

    return Matrix(map);
}

template <typename Scalar, typename MatProd>
std::tuple<EigenVector<Scalar>, EigenMatrix<Scalar>, int> eigsCompute(SymEigsSolver<MatProd> &eigs, size_t max_iter)
{
    eigs.init();
    eigs.compute(SortRule::LargestAlge, max_iter);

    EigenVector<Scalar> evalues;
    EigenMatrix<Scalar> evectors;
    if (eigs.info() == CompInfo::Successful)
    {
        evalues = eigs.eigenvalues();
        evectors = eigs.eigenvectors();
    }
    int status = static_cast<int>(eigs.info());

    return std::make_tuple(evalues, evectors, status);
}

template <typename Scalar, typename MatProd, typename InMatrix = Eigen::Ref<const EigenMatrix<Scalar>>>
std::tuple<EigenVector<Scalar>, EigenMatrix<Scalar>, int> eigs_eigen(InMatrix v, size_t nev, size_t ncv, size_t max_iter)
{
    MatProd op(v);
    SymEigsSolver<MatProd> eigs(op, nev, ncv);
    return eigsCompute<Scalar, MatProd>(eigs, max_iter);
}

template <typename Scalar, typename MatProd, typename InMatrix = ndarray<Scalar>>
std::tuple<EigenVector<Scalar>, EigenMatrix<Scalar>, int> eigs_pybackend(InMatrix v, size_t nev, size_t ncv, size_t max_iter, py::object backend)
{
    MatProd op(v, backend);
    SymEigsSolver<MatProd> eigs(op, nev, ncv);
    return eigsCompute<Scalar, MatProd>(eigs, max_iter);
}

template <typename Scalar, typename Vector = EigenVector<Scalar>, typename Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
std::tuple<Matrix, Vector, Matrix> partial_svd(Eigen::Ref<const Matrix> mat, size_t nev, size_t ncv)
{
    PartialSVDSolver<Matrix> svd(mat, nev, ncv);
    svd.compute();

    auto u = svd.matrix_U(nev);
    auto s = svd.singular_values();
    auto v = svd.matrix_V(nev);

    return std::make_tuple(u, s, v);
}

PYBIND11_MODULE(spectra_ext, m)
{
    // m.def("sym_eigs_dense_eigen_float32", &eigs_eigen<float, DenseSymMatProd<float>>);
    // m.def("sym_eigs_dense_eigen_float64", &eigs_eigen_sym<double, DenseSymMatProd<double>>);

    // m.def("eigs_sym_sparse_float32", &eigs_eigen_sym<float, SparseSymMatProd<float>, Eigen::SparseMatrix<float>>);
    // m.def("eigs_sym_sparse_float64", &eigs_eigen_sym<float, SparseSymMatProd<float>, Eigen::SparseMatrix<float>>);

    // Function based on python backends: float32/float64, dense/sparse
    m.def("sym_eigs_dense_pybackend_float32", &eigs_pybackend<float, PythonDenseSymMatProd<float>>);
    // m.def("sym_eigs_dense_pybackend_float64", &eigs_python_backend<double, PythonDenseSymMatProd<double>>);

    m.def("sym_eigs_sparse_pybackend_float32", &eigs_pybackend<float, PythonSparseSymMatProd<float, Eigen::SparseMatrix<float>>, Eigen::SparseMatrix<float>>);
    // m.def("sym_eigs_sparse_pybackend_float64", &eigs_pybackend<double, PythonSparseSymMatProd<double, Eigen::SparseMatrix<double>>, Eigen::SparseMatrix<double>>);

    // m.def("partial_svd_float32", &partial_svd<float>);
    // m.def("partial_svd_float64", &partial_svd<double>);
}
