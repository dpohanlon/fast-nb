#pragma once

#include <omp.h>

#include <Eigen/Dense>
#include <boost/sort/sort.hpp>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <cmath>

#include "cache.hpp"
#include "utils.hpp"

// Precompute log combinatorial coefficients for small values of k, r.
// This is only constexpr in GCC, but will be for both in C++26
#if defined(__GNUC__) && !defined(__clang__)
// #include <omp.h>
#include "log_comb_gcc.hpp"
#else
#include "log_comb.hpp"
#endif

// The 'lossless' optimisations versions

template <typename T>
double nb_base(int k, T r, double p) {
    if (k < 0) {
        return 0.0;
    }

    const double log_p = std::log(p);
    const double log_1_minus_p = std::log(1.0 - p);

    // Use log-gamma for binomial coefficient: log(choose(k + r - 1, r - 1))
    // TODO: Specialise separately for real and integer r
    double log_comb = compute_log_comb(k, r);

    // Compute log PMF and exponentiate
    return std::exp(log_comb + k * log_1_minus_p + r * log_p);
}

template <typename T>
double nb_base_fixed_r(int k, T r, double p, double lgamma_r) {
    if (k < 0) {
        return 0.0;
    }

    const double log_p = std::log(p);
    const double log_1_minus_p = std::log(1.0 - p);

    // Use log-gamma for binomial coefficient: log(choose(k + r - 1, r - 1))
    // TODO: Specialise separately for real and integer r
    double log_comb = compute_log_comb(k, r, lgamma_r);

    // Compute log PMF and exponentiate
    return std::exp(log_comb + k * log_1_minus_p + r * log_p);
}

template <>
double nb_base_fixed_r(int k, int r, double p, double lgamma_r) {
    if (k < 0) {
        return 0.0;
    }

    const double log_p = std::log(p);
    const double log_1_minus_p = std::log(1.0 - p);

    // Use log-gamma for binomial coefficient: log(choose(k + r - 1, r - 1))
    // TODO: Specialise separately for real and integer r
    double log_comb = compute_log_comb(k, r, lgamma_r);

    // Compute log PMF and exponentiate
    return std::exp(log_comb + k * log_1_minus_p + r * log_p);
}

double nb_base_fixed_r_opt(int k, double r, double p, double lgamma_r,
                           LgammaCache &lgamma_kr, LgammaCache &lgamma_k1,
                           bool log = false) {
    if (k < 0) {
        return 0.0;
    }

    const double log_p = std::log(p);
    const double log_1_minus_p = std::log(1.0 - p);

    double lg_kr = lgamma_kr.lgamma(k + r);
    double lg_k1 = lgamma_k1.lgamma(k + 1);

    double log_comb = lg_kr - lgamma_r - lg_k1;

    double res = log_comb + k * log_1_minus_p + r * log_p;

    return log ? res : std::exp(res);
}

double nb2_base(int k, double m, double r) {
    // m : mean
    // r : concentration

    double p = prob(m, r);

    return nb_base(k, r, p);
}

template <typename T>
double zinb2_base(T k, double m, double r, double alpha) {
    // m : mean
    // r : concentration

    double p = prob(m, r);
    double nb_prob = nb_base(k, r, p);

    return (k == 0) ? ((1.0 - alpha) * nb_prob + alpha) : ((1.0 - alpha) * nb_prob);
}
