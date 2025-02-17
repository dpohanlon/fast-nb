#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "eigen_nb.hpp"
#include "eigen_nb_jac.hpp"
#include "omp.h"

// TODO: Clean these up for generic function inputs

std::pair<double, double> optimise(Eigen::VectorXi& k, double m = 10.,
                                   double r = 10., double learning_rate = 0.1,
                                   int max_iterations = 1000) {
    const double tolerance = 1e-6;

    for (int iter = 0; iter < max_iterations; ++iter) {
        Eigen::MatrixXd grad_matrix =
            log_nb2_gradient_vec_eigen_blocks_no_copy(k, m, r);
        Eigen::Vector2d grad = -grad_matrix.colwise().mean();

        // TO DO: Pick a nice value of this to break early on
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

std::tuple<double, double, double> optimise_zi(Eigen::VectorXi& k,
                                               double m = 10., double r = 10.,
                                               double alpha = 0.1,
                                               double learning_rate = 0.1,
                                               int max_iterations = 1000) {
    const double tolerance = 1e-6;

    for (int iter = 0; iter < max_iterations; ++iter) {
        Eigen::MatrixXd grad_matrix =
            log_zinb_gradient_vec_eigen_blocks_post_process_select(k, m, r,
                                                                   alpha);
        Eigen::Vector2d grad = -grad_matrix.colwise().mean();

        m = m - learning_rate * grad[0];
        r = r - learning_rate * grad[1];
        alpha = r - learning_rate * grad[2];

        m = std::max(m, 1.0);
        r = std::max(r, 1e-8);
        alpha = std::clamp(alpha, 0.0, 1.0);
    }

    return std::make_tuple(m, r, alpha);
}

std::vector<std::pair<double, double>> optimise_all_genes(
    Eigen::MatrixXi& k, double m = 10., double r = 10.,
    double learning_rate = 1E-2, int max_iterations = 1000) {
    // Assume that each row corresponds to one gene.
    const int num_genes = k.rows();
    std::vector<std::pair<double, double>> optimized_params(num_genes);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_genes; ++i) {
        Eigen::VectorXi gene_data = k.row(i).transpose();
        optimized_params[i] =
            optimise(gene_data, m, r, learning_rate, max_iterations);
    }

    return optimized_params;
}

std::vector<std::tuple<double, double, double>> optimise_all_genes_zi(
    Eigen::MatrixXi& k, double m = 10., double r = 10., double alpha = 0.1,
    double learning_rate = 1E-2, int max_iterations = 1000) {
    // Assume that each row corresponds to one gene.
    const int num_genes = k.rows();
    std::vector<std::tuple<double, double, double>> optimized_params(num_genes);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_genes; ++i) {
        Eigen::VectorXi gene_data = k.row(i).transpose();
        optimized_params[i] =
            optimise_zi(gene_data, m, r, alpha, learning_rate, max_iterations);
    }

    return optimized_params;
}
