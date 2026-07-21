#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "eigen_nb.hpp"
#include "eigen_nb_jac.hpp"
#include "omp.h"

inline double sigmoid(double x) {
    if (x >= 0.0) {
        const double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(x);
    return z / (1.0 + z);
}

inline double softplus(double x) {
    if (x > 30.0) return x;
    if (x < -30.0) return std::exp(x);
    return std::log1p(std::exp(x));
}

inline double inv_softplus(double y) {
    y = std::max(y, 1e-12);
    if (y > 30.0) return y;
    return std::log(std::expm1(y));
}

inline double logit(double p) {
    p = std::clamp(p, 1e-10, 1.0 - 1e-10);
    return std::log(p) - std::log1p(-p);
}

inline bool converged_step(const Eigen::VectorXd &gradient,
                           const Eigen::VectorXd &step,
                           double gradient_tolerance = 1e-8,
                           double step_tolerance = 1e-10) {
    return gradient.lpNorm<Eigen::Infinity>() < gradient_tolerance ||
           step.lpNorm<Eigen::Infinity>() < step_tolerance;
}

inline std::pair<double, double> optimise(Eigen::VectorXi &k,
                                          double m = 10.0,
                                          double r = 10.0,
                                          double learning_rate = 0.1,
                                          int max_iterations = 1000) {
    if (k.size() == 0) {
        throw std::invalid_argument("Input count vector is empty.");
    }

    const double eps_m = 1e-12;
    const double eps_r = 1e-10;

    double theta_m = inv_softplus(std::max(m, eps_m));
    double theta_r = inv_softplus(std::max(r, eps_r));

    for (int iter = 0; iter < max_iterations; ++iter) {
        const double m_cur = softplus(theta_m) + eps_m;
        const double r_cur = softplus(theta_r) + eps_r;

        const auto grad_means =
            log_nb2_gradient_means_eigen_blocks(k, m_cur, r_cur);
        if (grad_means.second || !grad_means.first.allFinite()) {
            std::cerr << "Error: non-finite NB2 gradient." << std::endl;
            break;
        }

        // Minimise negative mean log-likelihood in unconstrained coordinates.
        Eigen::Vector2d gradient;
        gradient[0] = -grad_means.first[0] * sigmoid(theta_m);
        gradient[1] = -grad_means.first[1] * sigmoid(theta_r);

        const Eigen::Vector2d step = learning_rate * gradient;
        theta_m -= step[0];
        theta_r -= step[1];

        if (!std::isfinite(theta_m) || !std::isfinite(theta_r)) {
            std::cerr << "Error: non-finite NB2 optimizer state." << std::endl;
            break;
        }

        if (converged_step(gradient, step)) break;
    }

    return {softplus(theta_m) + eps_m, softplus(theta_r) + eps_r};
}

inline std::tuple<double, double, double> optimise_zi(
    Eigen::VectorXi &k, double m = 10.0, double r = 10.0,
    double alpha = 0.1, double learning_rate = 0.1,
    int max_iterations = 1000) {
    if (k.size() == 0) {
        throw std::invalid_argument("Input count vector is empty.");
    }

    const double eps_m = 1e-12;
    const double eps_r = 1e-10;
    const double eps_a = 1e-10;

    double theta_m = inv_softplus(std::max(m, eps_m));
    double theta_r = inv_softplus(std::max(r, eps_r));
    double theta_a = logit(std::clamp(alpha, eps_a, 1.0 - eps_a));

    for (int iter = 0; iter < max_iterations; ++iter) {
        const double m_cur = softplus(theta_m) + eps_m;
        const double r_cur = softplus(theta_r) + eps_r;
        const double a_cur =
            std::clamp(sigmoid(theta_a), eps_a, 1.0 - eps_a);

        const auto grad_means =
            log_zinb_gradient_means_eigen_blocks_post_process_select(
                k, m_cur, r_cur, a_cur);
        if (grad_means.second || !grad_means.first.allFinite()) {
            std::cerr << "Error: non-finite ZINB gradient." << std::endl;
            break;
        }

        Eigen::Vector3d gradient;
        gradient[0] = -grad_means.first[0] * sigmoid(theta_m);
        gradient[1] = -grad_means.first[1] * sigmoid(theta_r);
        gradient[2] =
            -grad_means.first[2] * a_cur * (1.0 - a_cur);

        const Eigen::Vector3d step = learning_rate * gradient;
        theta_m -= step[0];
        theta_r -= step[1];
        theta_a -= step[2];

        if (!std::isfinite(theta_m) || !std::isfinite(theta_r) ||
            !std::isfinite(theta_a)) {
            std::cerr << "Error: non-finite ZINB optimizer state."
                      << std::endl;
            break;
        }

        if (converged_step(gradient, step)) break;
    }

    return {softplus(theta_m) + eps_m, softplus(theta_r) + eps_r,
            std::clamp(sigmoid(theta_a), eps_a, 1.0 - eps_a)};
}

inline std::pair<double, double> optimise_exposure(
    Eigen::VectorXi &k, const Eigen::VectorXd &exposure,
    double mu0 = 10.0, double r = 10.0,
    double learning_rate = 0.1, int max_iterations = 1000) {
    if (k.size() == 0) {
        throw std::invalid_argument("Input count vector is empty.");
    }
    if (exposure.size() != k.size()) {
        throw std::invalid_argument(
            "Exposure length must equal the number of observations.");
    }
    if ((exposure.array() <= 0.0).any()) {
        throw std::invalid_argument("All exposure values must be positive.");
    }

    const double eps_mu = 1e-12;
    const double eps_r = 1e-10;

    double theta_mu = inv_softplus(std::max(mu0, eps_mu));
    double theta_r = inv_softplus(std::max(r, eps_r));

    for (int iter = 0; iter < max_iterations; ++iter) {
        const double mu_cur = softplus(theta_mu) + eps_mu;
        const double r_cur = softplus(theta_r) + eps_r;

        const auto grad_means = log_nb2_gradient_means_eigen_exposure(
            k, mu_cur, r_cur, exposure);
        if (grad_means.second || !grad_means.first.allFinite()) {
            std::cerr << "Error: non-finite exposure-NB gradient."
                      << std::endl;
            break;
        }

        Eigen::Vector2d gradient;
        gradient[0] = -grad_means.first[0] * sigmoid(theta_mu);
        gradient[1] = -grad_means.first[1] * sigmoid(theta_r);

        const Eigen::Vector2d step = learning_rate * gradient;
        theta_mu -= step[0];
        theta_r -= step[1];

        if (!std::isfinite(theta_mu) || !std::isfinite(theta_r)) {
            std::cerr << "Error: non-finite exposure-NB optimizer state."
                      << std::endl;
            break;
        }

        if (converged_step(gradient, step)) break;
    }

    return {softplus(theta_mu) + eps_mu, softplus(theta_r) + eps_r};
}

inline std::pair<std::vector<double>, std::vector<double>> optimise_all_genes(
    Eigen::MatrixXi &k, Eigen::VectorXd &m_vec, Eigen::VectorXd &r_vec,
    double learning_rate = 1e-2, int max_iterations = 1000) {
    const int num_genes = k.rows();

    if (m_vec.size() != num_genes || r_vec.size() != num_genes) {
        throw std::invalid_argument(
            "Size of m_vec and r_vec must equal the number of genes.");
    }

    std::vector<double> m_opt(num_genes);
    std::vector<double> r_opt(num_genes);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_genes; ++i) {
        Eigen::VectorXi gene_data = k.row(i).transpose();
        std::tie(m_opt[i], r_opt[i]) =
            optimise(gene_data, m_vec[i], r_vec[i], learning_rate,
                     max_iterations);
    }

    return {m_opt, r_opt};
}

inline std::tuple<std::vector<double>, std::vector<double>,
                  std::vector<double>>
optimise_all_genes_zi(Eigen::MatrixXi &k, std::vector<double> &m_vec,
                      std::vector<double> &r_vec,
                      std::vector<double> &alpha_vec,
                      double learning_rate = 1e-2,
                      int max_iterations = 1000) {
    const int num_genes = k.rows();

    if (m_vec.size() != static_cast<size_t>(num_genes) ||
        r_vec.size() != static_cast<size_t>(num_genes) ||
        alpha_vec.size() != static_cast<size_t>(num_genes)) {
        throw std::invalid_argument(
            "Size of m_vec, r_vec and alpha_vec must equal the number of genes.");
    }

    std::vector<double> m_opt(num_genes);
    std::vector<double> r_opt(num_genes);
    std::vector<double> a_opt(num_genes);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_genes; ++i) {
        Eigen::VectorXi gene_data = k.row(i).transpose();
        std::tie(m_opt[i], r_opt[i], a_opt[i]) =
            optimise_zi(gene_data, m_vec[i], r_vec[i], alpha_vec[i],
                        learning_rate, max_iterations);
    }

    return {m_opt, r_opt, a_opt};
}

inline std::pair<std::vector<double>, std::vector<double>>
optimise_all_genes_exposure(Eigen::MatrixXi &k, Eigen::VectorXd &m0_vec,
                            Eigen::VectorXd &r_vec,
                            const Eigen::VectorXd &exposure,
                            double learning_rate = 1e-2,
                            int max_iterations = 1000) {
    const int num_genes = k.rows();

    if (m0_vec.size() != num_genes || r_vec.size() != num_genes) {
        throw std::invalid_argument(
            "Size of m0_vec and r_vec must equal the number of genes.");
    }
    if (exposure.size() != k.cols()) {
        throw std::invalid_argument(
            "Exposure length must equal the number of cells.");
    }

    std::vector<double> m0_opt(num_genes);
    std::vector<double> r_opt(num_genes);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_genes; ++i) {
        Eigen::VectorXi gene_data = k.row(i).transpose();
        std::tie(m0_opt[i], r_opt[i]) =
            optimise_exposure(gene_data, exposure, m0_vec[i], r_vec[i],
                              learning_rate, max_iterations);
    }

    return {m0_opt, r_opt};
}
