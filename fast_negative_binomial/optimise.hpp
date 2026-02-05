#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "eigen_nb.hpp"
#include "eigen_nb_jac.hpp"
#include "omp.h"

// TODO: Clean these up for generic function inputs

double sigmoid(double x) {
    if (x >= 0.0) {
        double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    } else {
        double z = std::exp(x);
        return z / (1.0 + z);
    }
};

double softplus(double x) {
    if (x > 30.0) return x;
    if (x < -30.0) return std::exp(x);
    return std::log1p(std::exp(x));
};

double inv_softplus(double y) {
    if (y > 30.0) return y;
    return std::log(std::expm1(y));
};

double logit(double p) {
    p = std::clamp(p, 1e-8, 1.0 - 1e-8);
    return std::log(p) - std::log(1.0 - p);
};

std::pair<double, double> optimise(Eigen::VectorXi& k, double m = 10.,
                                   double r = 10., double learning_rate = 0.1,
                                   int max_iterations = 1000) {
    if (k.size() == 0) {
        std::cerr << "Error: input vector k is empty." << std::endl;
        return std::make_pair(m, r);
    }

    const double eps_m = 1e-12;
    const double eps_r = 1e-8;

    m = std::max(m, eps_m);
    r = std::max(r, eps_r);

    double theta_m = inv_softplus(m);
    double theta_r = inv_softplus(r);

    for (int iter = 0; iter < max_iterations; ++iter) {
        double m_cur = softplus(theta_m) + eps_m;
        double r_cur = softplus(theta_r) + eps_r;

        Eigen::MatrixXd grad_matrix =
            log_nb2_gradient_vec_eigen_blocks_no_copy(k, m_cur, r_cur);

        if (!grad_matrix.allFinite()) {
            std::cerr << "Error: Non-finite values encountered in grad_matrix." << std::endl;
            break;
        }

        Eigen::Vector2d g = -grad_matrix.colwise().mean(); // dL/dm, dL/dr

        double dm_dtheta = sigmoid(theta_m); // d softplus / dtheta
        double dr_dtheta = sigmoid(theta_r);

        theta_m = theta_m - learning_rate * g[0] * dm_dtheta;
        theta_r = theta_r - learning_rate * g[1] * dr_dtheta;

        if (!std::isfinite(theta_m) || !std::isfinite(theta_r)) {
            std::cerr << "Error: Non-finite theta values encountered." << std::endl;
            break;
        }
    }

    double m_out = softplus(theta_m) + eps_m;
    double r_out = softplus(theta_r) + eps_r;
    return std::make_pair(m_out, r_out);
}


std::tuple<double, double, double> optimise_zi(Eigen::VectorXi& k,
                                               double m = 10., double r = 10.,
                                               double alpha = 0.1,
                                               double learning_rate = 0.1,
                                               int max_iterations = 1000) {

    const double eps_m = 1e-12;
    const double eps_r = 1e-6;
    const double eps_a = 1e-8;

    m = std::max(m, eps_m);
    r = std::max(r, eps_r);
    alpha = std::clamp(alpha, eps_a, 1.0 - eps_a);

    double theta_m = inv_softplus(m);
    double theta_r = inv_softplus(r);
    double theta_a = logit(alpha);

    for (int iter = 0; iter < max_iterations; ++iter) {
        double m_cur = softplus(theta_m) + eps_m;
        double r_cur = softplus(theta_r) + eps_r;
        double a_cur = sigmoid(theta_a);
        a_cur = std::clamp(a_cur, eps_a, 1.0 - eps_a);

        Eigen::MatrixXd grad_matrix =
            log_zinb_gradient_vec_eigen_blocks_post_process_select(k, m_cur, r_cur, a_cur);

        if (!grad_matrix.allFinite()) {
            std::cerr << "Error: Non-finite values encountered in ZI grad_matrix." << std::endl;
            break;
        }

        Eigen::Vector3d g = -grad_matrix.colwise().mean(); // dL/dm, dL/dr, dL/dalpha

        double dm_dtheta = sigmoid(theta_m);
        double dr_dtheta = sigmoid(theta_r);
        double da_dtheta = a_cur * (1.0 - a_cur);

        theta_m = theta_m - learning_rate * g[0] * dm_dtheta;
        theta_r = theta_r - learning_rate * g[1] * dr_dtheta;
        theta_a = theta_a - learning_rate * g[2] * da_dtheta;

        if (!std::isfinite(theta_m) || !std::isfinite(theta_r) || !std::isfinite(theta_a)) {
            std::cerr << "Error: Non-finite theta values encountered in ZI update." << std::endl;
            break;
        }
    }

    double m_out = softplus(theta_m) + eps_m;
    double r_out = softplus(theta_r) + eps_r;
    double a_out = sigmoid(theta_a);
    a_out = std::clamp(a_out, eps_a, 1.0 - eps_a);

    return std::make_tuple(m_out, r_out, a_out);
}

std::pair<double, double> optimise_exposure(
    Eigen::VectorXi& k,
    const Eigen::VectorXd& exposure,
    double mu0 = 10.,
    double r = 10.,
    double learning_rate = 0.1,
    int max_iterations = 1000
) {
    for (int iter = 0; iter < max_iterations; ++iter) {
        Eigen::MatrixXd grad_matrix = log_nb2_gradient_vec_eigen_exposure(k, mu0, r, exposure);
        Eigen::Vector2d grad = -grad_matrix.colwise().mean();

        mu0 -= learning_rate * grad[0];
        r   -= learning_rate * grad[1];

        if (!std::isfinite(mu0) || !std::isfinite(r)) break;
        mu0 = std::max(mu0, 1.0);
        r   = std::max(r, 1e-8);
    }
    return std::make_pair(mu0, r);
}

std::pair<std::vector<double>, std::vector<double>> optimise_all_genes(
    Eigen::MatrixXi& k, Eigen::VectorXd& m_vec,
    Eigen::VectorXd& r_vec, double learning_rate = 1E-2,
    int max_iterations = 1000) {

    const int num_genes = k.rows();

    if (m_vec.size() != num_genes || r_vec.size() != num_genes) {
        throw std::invalid_argument("Size of m_vec and r_vec must equal the number of genes (rows in k).");
    }

    std::vector<double> m_opt(num_genes);
    std::vector<double> r_opt(num_genes);

    for (int i = 0; i < num_genes; ++i) {
        Eigen::VectorXi gene_data = k.row(i).transpose();
        std::tie(m_opt[i], r_opt[i]) =
            optimise(gene_data, m_vec[i], r_vec[i], learning_rate, max_iterations);
    }

    return std::make_pair(m_opt, r_opt);
}



std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> optimise_all_genes_zi(
    Eigen::MatrixXi& k, std::vector<double>& m_vec,
    std::vector<double>& r_vec, std::vector<double>& alpha_vec,
    double learning_rate = 1E-2, int max_iterations = 1000) {

    const int num_genes = k.rows();

    if (m_vec.size() != num_genes || r_vec.size() != num_genes || alpha_vec.size() != num_genes) {
        throw std::invalid_argument("Size of m_vec, r_vec, and alpha_vec must equal the number of genes (rows in k).");
    }

    std::vector<double> m_opt(num_genes);
    std::vector<double> r_opt(num_genes);
    std::vector<double> a_opt(num_genes);

    for (int i = 0; i < num_genes; ++i) {
        Eigen::VectorXi gene_data = k.row(i).transpose();
        std::tie(m_opt[i], r_opt[i], a_opt[i]) =
            optimise_zi(gene_data, m_vec[i], r_vec[i], alpha_vec[i], learning_rate, max_iterations);
    }

    return std::make_tuple(m_opt, r_opt, a_opt);
}


std::pair<std::vector<double>, std::vector<double>> optimise_all_genes_exposure(
    Eigen::MatrixXi& k,                 // shape: genes x cells
    Eigen::VectorXd& m0_vec,            // mu0 per gene
    Eigen::VectorXd& r_vec,             // r per gene
    const Eigen::VectorXd& exposure,    // length = cells
    double learning_rate = 1E-2,
    int max_iterations = 1000
) {
    const int num_genes = k.rows();
    if (m0_vec.size() != num_genes || r_vec.size() != num_genes) {
        throw std::invalid_argument("Size of m0_vec and r_vec must equal number of genes.");
    }
    if (exposure.size() != k.cols()) {
        throw std::invalid_argument("Exposure length must equal number of cells (k.cols()).");
    }

    std::vector<double> r_opt(num_genes);
    std::vector<double> m0_opt(num_genes);

    for (int i = 0; i < num_genes; ++i) {
        Eigen::VectorXi gene_data = k.row(i).transpose();
        auto pr = optimise_exposure(gene_data, exposure, m0_vec[i], r_vec[i], learning_rate, max_iterations);
        m0_opt[i] = pr.first;
        r_opt[i]  = pr.second;
    }
    return std::make_pair(m0_opt, r_opt);
}
