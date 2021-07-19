// Almost exact copy from Spectra/contrib/PartialSVDSolver.h but with python func calling
#include <Eigen/Core>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/contrib/PartialSVDSolver.h>

using namespace std;
using namespace Spectra;

namespace py = pybind11;


template <typename Scalar, typename MatrixType>
class SVDTallMatOpPython : public SVDMatOp<Scalar>
{
private:
    using Index = Eigen::Index;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MapConstVec = Eigen::Map<const Vector>;
    using MapVec = Eigen::Map<Vector>;
    using ConstGenericMatrix = const Eigen::Ref<const MatrixType>;

    ConstGenericMatrix m_mat;
    const Index m_dim;
    py::object backend;

public:
    SVDTallMatOpPython(ConstGenericMatrix& mat, py::object backend) :
        m_mat(mat),
        m_dim((std::min)(mat.rows(), mat.cols()))
    {
        this->backend = backend(mat);
    }

    // These are the rows and columns of A' * A
    Index rows() const override { return m_dim; }
    Index cols() const override { return m_dim; }

    // y_out = A' * A * x_in
    void perform_op(const Scalar* x_in, Scalar* y_out) const override
    {
        MapConstVec x(x_in, m_mat.cols());
        MapVec y(y_out, m_mat.cols());
        this->backend.attr("perform_op")(x, y);
    }
};

template <typename Scalar, typename MatrixType>
class SVDWideMatOpPython : public SVDMatOp<Scalar>
{
private:
    using Index = Eigen::Index;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MapConstVec = Eigen::Map<const Vector>;
    using MapVec = Eigen::Map<Vector>;
    using ConstGenericMatrix = const Eigen::Ref<const MatrixType>;

    ConstGenericMatrix m_mat;
    const Index m_dim;
    py::object backend;

public:
    SVDWideMatOpPython(ConstGenericMatrix& mat, py::object backend) :
        m_mat(mat),
        m_dim((std::min)(mat.rows(), mat.cols()))
    {
        this->backend = backend(mat.transpose());
    }

    // These are the rows and columns of A * A'
    Index rows() const override { return m_dim; }
    Index cols() const override { return m_dim; }

    // y_out = A * A' * x_in
    void perform_op(const Scalar* x_in, Scalar* y_out) const override
    {
        MapConstVec x(x_in, m_mat.rows());
        MapVec y(y_out, m_mat.rows());
        this->backend.attr("perform_op")(x, y);
    }
};

template <typename MatrixType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
class PythonPartialSVDSolver
{
private:
    using Scalar = typename MatrixType::Scalar;
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using ConstGenericMatrix = const Eigen::Ref<const MatrixType>;

    ConstGenericMatrix m_mat;
    const Index m_m;
    const Index m_n;
    SVDMatOp<Scalar> *m_op;
    SymEigsSolver<SVDMatOp<Scalar>> *m_eigs;
    Index m_nconv;
    Matrix m_evecs;

public:
    PythonPartialSVDSolver(ConstGenericMatrix &mat, Index ncomp, Index ncv, py::object backend) : m_mat(mat), m_m(mat.rows()), m_n(mat.cols()), m_evecs(0, 0)
    {
        // Determine the matrix type, tall or wide
        if (m_m > m_n)
        {
            m_op = new SVDTallMatOpPython<Scalar, MatrixType>(mat, backend);
        }
        else
        {
            m_op = new SVDWideMatOpPython<Scalar, MatrixType>(mat, backend);
        }

        m_eigs = new SymEigsSolver<SVDMatOp<Scalar>>(*m_op, ncomp, ncv);
    }

    virtual ~PythonPartialSVDSolver()
    {
        delete m_eigs;
        delete m_op;
    }

    Index compute(Index maxit = 1000, Scalar tol = 1e-10)
    {
        m_eigs->init();
        m_nconv = m_eigs->compute(SortRule::LargestAlge, maxit, tol);

        return m_nconv;
    }

    Vector singular_values() const
    {
        Vector svals = m_eigs->eigenvalues().cwiseSqrt();

        return svals;
    }

    Matrix matrix_U(Index nu)
    {
        if (m_evecs.cols() < 1)
        {
            m_evecs = m_eigs->eigenvectors();
        }
        nu = (std::min)(nu, m_nconv);
        if (m_m <= m_n)
        {
            return m_evecs.leftCols(nu);
        }

        return m_mat * (m_evecs.leftCols(nu).array().rowwise() / m_eigs->eigenvalues().head(nu).transpose().array().sqrt()).matrix();
    }

    Matrix matrix_V(Index nv)
    {
        if (m_evecs.cols() < 1)
        {
            m_evecs = m_eigs->eigenvectors();
        }
        nv = (std::min)(nv, m_nconv);
        if (m_m > m_n)
        {
            return m_evecs.leftCols(nv);
        }

        return m_mat.transpose() * (m_evecs.leftCols(nv).array().rowwise() / m_eigs->eigenvalues().head(nv).transpose().array().sqrt()).matrix();
    }
};
