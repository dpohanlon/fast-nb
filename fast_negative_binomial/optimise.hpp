#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "eigen_nb.hpp"
#include "eigen_nb_jac.hpp"
#include "omp.h"

// TODO: Clean these up for generic function inputs

std::pair<double, double> optimise(Eigen::VectorXi& k, double m = 10.,
                                   double r = 10., double learning_rate = 0.1,
                                   int max_iterations = 1000) {
    const double tolerance = 1e-6;

    if (k.size() == 0) {
        std::cerr << "Error: input vector k is empty." << std::endl;
        return std::make_pair(m, r);
    }

    for (int iter = 0; iter < max_iterations; ++iter) {
        Eigen::MatrixXd grad_matrix =
            log_nb2_gradient_vec_eigen_blocks_no_copy(k, m, r);

        if (!grad_matrix.allFinite()) {
            std::cerr << "Error: Non-finite values encountered in grad_matrix." << std::endl;
            break;
        }

        Eigen::Vector2d grad = -grad_matrix.colwise().mean();

        // TO DO: Pick a nice value of this to break early on
        // if (grad.norm() < tolerance) {
        //     break;
        // }

        m = m - learning_rate * grad[0];
        r = r - learning_rate * grad[1];

        if (!std::isfinite(m) || !std::isfinite(r)) {
            std::cerr << "Error: Non-finite parameter values encountered before update." << " " << m << " " << r << std::endl;
            break;
        }

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

std::pair<std::vector<double>, std::vector<double>> optimise_all_genes(
    Eigen::MatrixXi& k,  Eigen::VectorXd& m_vec,
    Eigen::VectorXd& r_vec, double learning_rate = 1E-2,
    int max_iterations = 1000) {

    const int num_genes = k.rows();

    if (m_vec.size() != num_genes || r_vec.size() != num_genes) {
        throw std::invalid_argument("Size of m_vec and r_vec must equal the number of genes (rows in k).");
    }

    std::vector<double> r_opt(num_genes);
    std::vector<double> m_opt(num_genes);

    // Don't parallelise here, as it's done elsewhere within the loop
    for (int i = 0; i < num_genes; ++i) {
        Eigen::VectorXi gene_data = k.row(i).transpose();
        std::tie(r_opt[i], m_opt[i]) = optimise(gene_data, m_vec[i], r_vec[i], learning_rate, max_iterations);
    }

    return std::make_pair(r_opt, m_opt);
}


std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> optimise_all_genes_zi(
    Eigen::MatrixXi& k, std::vector<double>& m_vec,
    std::vector<double>& r_vec, std::vector<double>& alpha_vec,
    double learning_rate = 1E-2, int max_iterations = 1000) {

    const int num_genes = k.rows();

    if (m_vec.size() != num_genes || r_vec.size() != num_genes || alpha_vec.size() != num_genes) {
        throw std::invalid_argument("Size of m_vec, r_vec, and alpha_vec must equal the number of genes (rows in k).");
    }

    std::vector<double> r_opt(num_genes);
    std::vector<double> m_opt(num_genes);
    std::vector<double> a_opt(num_genes);

    for (int i = 0; i < num_genes; ++i) {
        Eigen::VectorXi gene_data = k.row(i).transpose();
        std::tie(r_opt[i], m_opt[i], a_opt[i]) = optimise_zi(gene_data, m_vec[i], r_vec[i], alpha_vec[i], learning_rate, max_iterations);
    }

    return std::make_tuple(r_opt, m_opt, a_opt);
}
