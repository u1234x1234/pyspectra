/*
<%
setup_pybind11(cfg)
%>
*/
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <iostream>
#include <Eigen/Core>
#include <Spectra/SymEigsSolver.h>

using namespace std;
using namespace Spectra;

Eigen::VectorXd eigs(Eigen::Ref<Eigen::MatrixXd> v)
{
    DenseSymMatProd<double> op(v);
    SymEigsSolver<DenseSymMatProd<double>> eigs(op, 3, 6);

    eigs.init();
    int nconv = eigs.compute(SortRule::LargestAlge);

    Eigen::VectorXd evalues;
    if(eigs.info() == CompInfo::Successful)
        evalues = eigs.eigenvalues();

    return evalues;
}

PYBIND11_MODULE(spectra_ext, m)
{
    m.def("eigs", &eigs);
}
