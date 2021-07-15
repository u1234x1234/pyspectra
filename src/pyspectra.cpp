/*
<%
setup_pybind11(cfg)
%>
*/
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include <iostream>
#include <Eigen/Core>
#include <Spectra/SymEigsSolver.h>

using namespace std;
using namespace Spectra;

namespace py = pybind11;

using np_array_f32 = pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>;
using np_array_f64 = pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>;


template <typename Scalar_, int Uplo = Eigen::Lower, int Flags = Eigen::ColMajor>
class MyDenseSymMatProd
{
public:
    using Scalar = Scalar_;

private:
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Flags>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MapConstVec = Eigen::Map<const Vector>;
    using MapVec = Eigen::Map<Vector>;
    // using ConstGenericMatrix = const Eigen::Ref<const Matrix>;

    // ConstGenericMatrix m_mat;
    py::buffer_info info;
    double *matData;
    py::object func;
    np_array_f64 array;

public:
    MyDenseSymMatProd(np_array_f64 array, py::object func)
    {
        this->array = array;
        info = array.request();
        matData = static_cast<double*>(info.ptr);
        // auto r = array.template mutable_unchecked<2>();
        // decltype(r)::foo= 1;
        this->func = func;
    }

    Index rows() const { return info.shape[0]; }
    Index cols() const { return info.shape[1]; }

    void perform_op(const Scalar* x_in, Scalar* y_out) const
    {
        MapConstVec x(x_in, info.shape[1]);
        MapVec y(y_out, info.shape[0]);
        // y.noalias() = m_mat.template selfadjointView<Uplo>() * x;
        // std::cout << 23 << " " << x_in[0] << " " x_in[2] << std::endl;
        // auto x_np = py::array_t<double>({200, 200}, info.strides, x_in);

        auto r = func(array, x, y);
    }

    Matrix operator*(const Eigen::Ref<const Matrix>& mat_in) const
    {
        std::cout << 12 << std::endl;
        // return m_mat.template selfadjointView<Uplo>() * mat_in;
    }

    Scalar operator()(Index i, Index j) const
    {
        return matData[i*200+j];
    }
};


template <typename num_t, typename Matrix>
Matrix ndarrayToMatrixX(pybind11::array_t<num_t, pybind11::array::c_style | pybind11::array::forcecast> b) {
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

// Eigen::VectorXd eigs(Eigen::Ref<Eigen::MatrixXd> v)
// {
//    DenseSymMatProd<double> op(v);
//     SymEigsSolver<DenseSymMatProd<double>> eigs(op, 64, 64*2);

//     eigs.init();
//     int nconv = eigs.compute(SortRule::LargestAlge);

//     Eigen::VectorXd evalues;
//     if(eigs.info() == CompInfo::Successful)
//         evalues = eigs.eigenvalues();

//     return evalues;
// }

Eigen::VectorXd eigs2(np_array_f64 v, py::object func)
{
    // Eigen::MatrixXd v = ndarrayToMatrixX<double, Eigen::MatrixXd>(v0);

    MyDenseSymMatProd<double> op(v, func);
    SymEigsSolver<MyDenseSymMatProd<double>> eigs(op, 3, 3*2);

    eigs.init();
    int nconv = eigs.compute(SortRule::LargestAlge);

    Eigen::VectorXd evalues;
    if(eigs.info() == CompInfo::Successful)
        evalues = eigs.eigenvalues();

    return evalues;
}


// using np_array_f32 = pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>;
// np_array_f32 eigs(std::function<np_array_f32(np_array_f32)> func, np_array_f32 x) {
//     return func(x);
// }


PYBIND11_MODULE(spectra_ext, m)
{
    // m.def("eigs", &eigs);
    m.def("eigs2", &eigs2);
    // m.def("np_to_eigen32", &np_to_eigen<float, Eigen::MatrixXf>);
    // m.def("np_to_eigen64", &np_to_eigen<double, Eigen::MatrixXd>);
}
