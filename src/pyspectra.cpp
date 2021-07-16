#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include <iostream>
#include <Eigen/Core>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/contrib/PartialSVDSolver.h>

using namespace std;
using namespace Spectra;

namespace py = pybind11;

template <typename T>
using ndarray = pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>;

template <typename T>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
template <typename T>
using EigenVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename Scalar_, int Uplo = Eigen::Lower, int Flags = Eigen::ColMajor>
class PythonSymMatProd
{
public:
    using Scalar = Scalar_;

private:
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Flags>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MapConstVec = Eigen::Map<const Vector>;
    using MapVec = Eigen::Map<Vector>;

    py::buffer_info info;
    py::object backend;

public:
    PythonSymMatProd(ndarray<Scalar_> array, py::object backend)
    {
        this->backend = backend(array);
        info = array.request();
    }

    Index rows() const { return info.shape[0]; }
    Index cols() const { return info.shape[1]; }

    void perform_op(const Scalar *x_in, Scalar *y_out) const
    {
        MapConstVec x(x_in, info.shape[1]);
        MapVec y(y_out, info.shape[0]);
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

template <typename Scalar, typename MatProd, typename Vector = EigenVector<Scalar>, typename Matrix = EigenMatrix<Scalar>>
std::tuple<Vector, Matrix, int> eigs_eigen_sym(Eigen::Ref<const Matrix> v, size_t nev, size_t ncv, size_t max_iter)
{
    MatProd op(v);
    SymEigsSolver<MatProd> eigs(op, nev, ncv);
    eigs.init();
    eigs.compute(SortRule::LargestAlge, max_iter);

    Vector evalues;
    Matrix evectors;
    if (eigs.info() == CompInfo::Successful) {
        evalues = eigs.eigenvalues();
        evectors = eigs.eigenvectors();
    }
    int status = static_cast<int>(eigs.info());

    return std::make_tuple(evalues, evectors, status);
}

template <typename Scalar, typename Vector = EigenVector<Scalar>, typename Matrix = EigenMatrix<Scalar>>
std::tuple<Vector, Matrix, int> eigs_python_backend(ndarray<Scalar> v, size_t nev, size_t ncv, size_t max_iter, py::object func)
{
    PythonSymMatProd<Scalar> op(v, func);
    SymEigsSolver<PythonSymMatProd<Scalar>> eigs(op, nev, ncv);

    eigs.init();
    eigs.compute(SortRule::LargestAlge, max_iter);

    Vector evalues;
    Matrix evectors;
    if (eigs.info() == CompInfo::Successful) {
        evalues = eigs.eigenvalues();
        evectors = eigs.eigenvectors();
    }
    int status = static_cast<int>(eigs.info());

    return std::make_tuple(evalues, evectors, status);
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
    m.def("eigs_sym_dense_float32", &eigs_eigen_sym<float, DenseSymMatProd<float>>);
    m.def("eigs_sym_dense_float64", &eigs_eigen_sym<double, DenseSymMatProd<double>>);

    m.def("eigs_python_backend_float32", &eigs_python_backend<float>);
    m.def("eigs_python_backend_float64", &eigs_python_backend<double>);

    m.def("partial_svd_float32", &partial_svd<float>);
    m.def("partial_svd_float64", &partial_svd<double>);
}
