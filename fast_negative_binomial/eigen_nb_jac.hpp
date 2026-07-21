#pragma once

#include <omp.h>

#include <Eigen/Dense>
#include <boost/math/special_functions/digamma.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

#include "eigen_nb.hpp"
#include "utils.hpp"

/*
 * Analytic score functions for the NB2 parameterisation
 *
 *   p = r / (r + m)
 *   E[K] = m
 *   Var[K] = m + m^2 / r
 *
 * For one observation k,
 *
 *   d log P / d m = k/m - (k+r)/(m+r)
 *
 *   d log P / d r = digamma(k+r) - digamma(r)
 *                     + log(r/(r+m)) + (m-k)/(m+r)
 *
 * The final term in d/dr is the chain-rule contribution from p(r,m).
 */

inline Eigen::Vector2d log_nb2_score_single(int k, double m, double r) {
    const double eps_m = 1e-12;
    const double eps_r = 1e-12;

    m = std::max(m, eps_m);
    r = std::max(r, eps_r);

    const double kr = static_cast<double>(k) + r;
    const double mr = m + r;

    Eigen::Vector2d score;
    score[0] = static_cast<double>(k) / m - kr / mr;
    score[1] = boost::math::digamma(kr) - boost::math::digamma(r) +
               std::log(r / mr) + (m - static_cast<double>(k)) / mr;
    return score;
}

inline Eigen::MatrixXd log_nb2_gradient_vec_eigen_blocks(
    const Eigen::VectorXi &k, double m, double r) {
    Eigen::MatrixXd out(k.size(), 2);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < k.size(); ++i) {
        out.row(i) = log_nb2_score_single(k[i], m, r).transpose();
    }

    return out;
}

inline Eigen::MatrixXd log_nb2_gradient_vec_eigen_blocks_no_copy(
    Eigen::Ref<Eigen::VectorXi> k, double m, double r) {
    return log_nb2_gradient_vec_eigen_blocks(k, m, r);
}

inline Eigen::MatrixXd log_nb_gradient_vec_eigen_blocks(
    const Eigen::VectorXi &k, double r, double p) {
    const double eps = 1e-12;
    p = std::clamp(p, eps, 1.0 - eps);
    r = std::max(r, eps);

    Eigen::MatrixXd out(k.size(), 2);
    const double digamma_r = boost::math::digamma(r);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < k.size(); ++i) {
        const double ki = static_cast<double>(k[i]);
        out(i, 0) = r / p - ki / (1.0 - p);
        out(i, 1) = std::log(p) - digamma_r +
                    boost::math::digamma(ki + r);
    }

    return out;
}

inline Eigen::MatrixXd log_nb_gradient_vec_eigen_blocks_no_copy(
    Eigen::Ref<Eigen::VectorXi> k, double r, double p) {
    return log_nb_gradient_vec_eigen_blocks(k, r, p);
}

inline std::pair<Eigen::Vector2d, bool>
log_nb_gradient_means_eigen_blocks_no_copy(Eigen::Ref<Eigen::VectorXi> k,
                                           double r, double p) {
    Eigen::MatrixXd grad = log_nb_gradient_vec_eigen_blocks(k, r, p);
    const bool bad = !grad.allFinite();
    return {grad.colwise().mean(), bad};
}

inline std::pair<Eigen::Vector2d, bool>
log_nb_gradient_means_eigen_blocks(const Eigen::VectorXi &k, double r,
                                   double p) {
    Eigen::VectorXi copy = k;
    return log_nb_gradient_means_eigen_blocks_no_copy(copy, r, p);
}

inline std::pair<Eigen::Vector2d, bool>
log_nb2_gradient_means_eigen_blocks(const Eigen::VectorXi &k, double m,
                                    double r) {
    Eigen::MatrixXd grad = log_nb2_gradient_vec_eigen_blocks(k, m, r);
    const bool bad = !grad.allFinite();
    return {grad.colwise().mean(), bad};
}

/*
 * ZINB score for
 *
 *   P(K=k) = (1-alpha) NB(k; m,r) + alpha I[k=0]
 *
 * alpha is the structural-zero probability.
 */
inline Eigen::Vector3d log_zinb_score_single(int k, double m, double r,
                                             double alpha) {
    const double eps_a = 1e-10;
    const double eps_q = 1e-300;

    alpha = std::clamp(alpha, eps_a, 1.0 - eps_a);
    m = std::max(m, 1e-12);
    r = std::max(r, 1e-12);

    const Eigen::Vector2d nb_score = log_nb2_score_single(k, m, r);
    Eigen::Vector3d score;

    if (k != 0) {
        score[0] = nb_score[0];
        score[1] = nb_score[1];
        score[2] = -1.0 / (1.0 - alpha);
        return score;
    }

    const double p = r / (r + m);
    const double q0 = std::max(std::pow(p, r), eps_q);
    const double denom = std::max(alpha + (1.0 - alpha) * q0, eps_q);

    // Posterior probability that an observed zero came from the NB component.
    const double nb_responsibility = (1.0 - alpha) * q0 / denom;

    score[0] = nb_responsibility * nb_score[0];
    score[1] = nb_responsibility * nb_score[1];
    score[2] = (1.0 - q0) / denom;
    return score;
}

inline Eigen::MatrixXd log_zinb_gradient_vec_eigen_blocks_post_process_select(
    const Eigen::VectorXi &k, double m, double r, double alpha) {
    Eigen::MatrixXd out(k.size(), 3);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < k.size(); ++i) {
        out.row(i) = log_zinb_score_single(k[i], m, r, alpha).transpose();
    }

    return out;
}

inline std::pair<Eigen::Vector3d, bool>
log_zinb_gradient_means_eigen_blocks_post_process_select(
    const Eigen::VectorXi &k, double m, double r, double alpha) {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    int bad = 0;

#pragma omp parallel
    {
        Eigen::Vector3d local = Eigen::Vector3d::Zero();
        int local_bad = 0;

#pragma omp for schedule(static) nowait
        for (int i = 0; i < k.size(); ++i) {
            const Eigen::Vector3d score =
                log_zinb_score_single(k[i], m, r, alpha);
            local += score;
            local_bad |= !score.allFinite();
        }

#pragma omp critical
        {
            sum += local;
            bad |= local_bad;
        }
    }

    if (k.size() == 0) {
        return {Eigen::Vector3d::Zero(), true};
    }

    return {sum / static_cast<double>(k.size()), bad != 0};
}

inline Eigen::Vector2d log_nb2_exposure_score_single(int k, double mu0,
                                                     double r,
                                                     double exposure) {
    const double eps = 1e-12;
    mu0 = std::max(mu0, eps);
    r = std::max(r, eps);
    exposure = std::max(exposure, eps);

    const double m = mu0 * exposure;
    const Eigen::Vector2d score_mr = log_nb2_score_single(k, m, r);

    Eigen::Vector2d out;
    out[0] = score_mr[0] * exposure;
    out[1] = score_mr[1];
    return out;
}

inline Eigen::MatrixXd log_nb2_gradient_vec_eigen_exposure(
    const Eigen::VectorXi &k, double mu0, double r,
    const Eigen::VectorXd &exposure) {
    if (exposure.size() != k.size()) {
        throw std::invalid_argument(
            "Exposure length must equal the number of observations.");
    }

    Eigen::MatrixXd out(k.size(), 2);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < k.size(); ++i) {
        out.row(i) =
            log_nb2_exposure_score_single(k[i], mu0, r, exposure[i])
                .transpose();
    }

    return out;
}

inline std::pair<Eigen::Vector2d, bool>
log_nb2_gradient_means_eigen_exposure(const Eigen::VectorXi &k, double mu0,
                                      double r,
                                      const Eigen::VectorXd &exposure) {
    Eigen::MatrixXd grad =
        log_nb2_gradient_vec_eigen_exposure(k, mu0, r, exposure);
    const bool bad = !grad.allFinite();
    return {grad.colwise().mean(), bad};
}
