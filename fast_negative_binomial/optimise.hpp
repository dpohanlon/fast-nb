#include <iostream>
#include <Eigen/Dense>
#include <algorithm>

#include "eigen_nb.hpp"
#include "eigen_nb_jac.hpp"

std::pair<double, double> optimise(Eigen::VectorX & k, double learning_rate = 1E-2, int max_iterations = 1000) {

    double m = 10.0;
    double r = 10.0;

    const double tolerance = 1e-6;

    for (int iter = 0; iter < max_iterations; ++iter) {
        Eigen::VectorXd log_vals = log_nb2_base_vec_eigen_blocks_no_copy(k, m, r);

        double total_log_lik = -log_vals.sum();

        Eigen::MatrixXd grad_matrix = log_nb2_gradient_vec_eigen_blocks_no_copy(k, m, r);
        Eigen::Vector2d grad = -grad_matrix.colwise().sum();

        if (grad.norm() < tolerance) {
            std::cout << "Converged at iteration " << iter << std::endl;
            break;
        }

        m = m - learning_rate * grad[0];
        r = r - learning_rate * grad[1];

        m = std::max(m, 1e-8);
        r = std::max(r, 1e-8);

        if (iter % 100 == 0) {
            std::cout << "Iteration " << iter
                      << ": Objective = " << objective
                      << ", m = " << m
                      << ", r = " << r << std::endl;
        }
    }

    std::cout << "Final optimized parameters:\n";
    std::cout << "m = " << m << "\n";
    std::cout << "r = " << r << std::endl;

    return std::make_pair(m, r);
}
