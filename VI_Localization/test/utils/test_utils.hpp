#include <Eigen/Core>

namespace testutils
{
    // Helper function to calculate the numerical Jacobian for a function// see L4S14/ Numeric Differentiation Methods
    template <typename Func>
    Eigen::MatrixXd calculateNumericalJacobian(Func func, const Eigen::VectorXd &x, double epsilon = 1e-6)
    {
        Eigen::MatrixXd J(func(x).size(), x.size());

        for (int i = 0; i < x.size(); ++i)
        {
            Eigen::VectorXd x_plus = x;
            Eigen::VectorXd x_minus = x;
            x_plus(i) += epsilon;
            x_minus(i) -= epsilon;
            J.col(i) = (func(x_plus) - func(x_minus)) / (2.0 * epsilon);
        }
        return J;
    }

} // namespace testutils
