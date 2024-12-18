#pragma once

#include <vector>
#include <cmath>
#include <numeric>

#include <iostream>
#include <cmath>
#include <limits>
#include <iomanip>
#include <map>
#include <tuple>

#include <cmath>

#include <Eigen/Dense>

#include <utils.hpp>

// Precompute log combinatorial coefficients for small values of k, r.
// This is only constexpr in GCC, but will be for both in C++26
#if defined(__GNUC__) && !defined(__clang__)
    #include <omp.h>
    #include "log_comb_gcc.hpp"
#else
    #include "log_comb.hpp"
#endif

// The 'lossless' optimisations versions

template<typename T>
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

double nb2_base(int k, double m, double r)
{
    // m : mean
    // r : concentration

    double p = prob(m, r);

    return nb_base(k, r, p);
}

template<typename T>
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

template<>
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

// I can probably template specialise these for vector or scalar, but not sure if it's worth it

template<typename T>
std::vector<double> nb_base_vec(std::vector<int> k, T r, double p)
{
    double lgamma_r = std::lgamma(static_cast<double>(r));

    std::vector<double> results(k.size());

    for (int i = 0; i < k.size(); ++i) {
        results[i] = nb_base_fixed_r(k[i], r, p, lgamma_r);
    }

    return results;
}

std::vector<double> nb2_base_vec(std::vector<int> k, double m, double r)
{
    // m : mean
    // r : concentration

    double p = prob(m, r);

    return nb_base_vec(k, r, p);
}

template<typename T>
Eigen::VectorXd nb_base_vec_eigen(const Eigen::VectorXi &k, T r, double p)
{
    double lgamma_r = std::lgamma(static_cast<double>(r));
    Eigen::VectorXd results(k.size());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < k.size(); ++i) {
        results[i] = nb_base_fixed_r(k[i], r, p, lgamma_r);
    }

    return results;
}
