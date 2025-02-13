#include <iostream>
#include <Eigen/Dense>
#include <algorithm>

#include "omp.h"

#include "eigen_nb.hpp"
#include "eigen_nb_jac.hpp"

std::pair<double, double> optimise(Eigen::VectorXi & k, double m = 10., double r = 10., double learning_rate = 0.1, int max_iterations = 1000) {

    const double tolerance = 1e-6;

    for (int iter = 0; iter < max_iterations; ++iter) {
        Eigen::VectorXd log_vals = log_nb2_base_vec_eigen_blocks_no_copy(k, m, r);

        double total_log_lik = -log_vals.sum();

        Eigen::MatrixXd grad_matrix = log_nb2_gradient_vec_eigen_blocks_no_copy(k, m, r);
        Eigen::Vector2d grad = -grad_matrix.colwise().mean();

        // if (grad.norm() < tolerance) {
        //     break;
        // }

        m = m - learning_rate * grad[0];
        r = r - learning_rate * grad[1];

        m = std::max(m, 1.0);
        r = std::max(r, 1e-8);

    }

    return std::make_pair(m, r);
}

std::vector<std::pair<double, double>> optimise_all_genes(Eigen::MatrixXi & k,
                                                            double m = 10.,
                                                            double r = 10.,
                                                            double learning_rate = 1E-2,
                                                            int max_iterations = 1000)
{
    // Assume that each row corresponds to one gene.
    const int num_genes = k.rows();
    std::vector<std::pair<double, double>> optimized_params(num_genes);

    # pragma omp parallel for schedule(static)
    for (int i = 0; i < num_genes; ++i) {
        Eigen::VectorXi gene_data = k.row(i).transpose();
        optimized_params[i] = optimise(gene_data, m, r, learning_rate, max_iterations);
    }

    return optimized_params;
}
