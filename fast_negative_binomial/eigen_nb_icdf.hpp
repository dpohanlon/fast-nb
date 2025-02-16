#pragma once

#include <omp.h>

#include <Eigen/Dense>
#include <boost/math/special_functions/beta.hpp>
#include <boost/sort/sort.hpp>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <cmath>
#include <numeric>

#include "eigen_nb_cdf.hpp"
#include "utils.hpp"

// Use a strategy of doubling and binary search between lo and new hi to find k
inline int nb_invcdf_single(double r, double p, double alpha) {
    if (alpha <= 0.0) return 0;
    if (alpha >= 1.0) return INT_MAX;

    int lo = 0;
    int hi = 1;
    while (nb_cdf_single(hi, r, p) < alpha && hi < (1 << 29)) {
        hi <<= 1;
    }

    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (nb_cdf_single(mid, r, p) >= alpha) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

// Parallel chunked version for a vector of alpha
Eigen::VectorXi nb_invcdf_vec_blocks(const Eigen::VectorXd &alpha_vec, double r,
                                     double p) {
    Eigen::VectorXi result(alpha_vec.size());
    int n = alpha_vec.size();
    int num_blocks = n / BLOCK_SIZE;
    int remainder = n % BLOCK_SIZE;

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int b = 0; b < num_blocks; ++b) {
            int start = b * BLOCK_SIZE;
            int end = start + BLOCK_SIZE;
            for (int i = start; i < end; ++i) {
                result(i) = nb_invcdf_single(r, p, alpha_vec(i));
            }
        }
    }

    // Handle any leftover elements
    int start = num_blocks * BLOCK_SIZE;
    for (int i = start; i < start + remainder; ++i) {
        result(i) = nb_invcdf_single(r, p, alpha_vec(i));
    }

    return result;
}

Eigen::VectorXi nb2_invcdf_vec(const Eigen::VectorXd &alpha_vec, double m,
                               double r) {
    double p = prob(m, r);
    return nb_invcdf_vec(alpha_vec, r, p);
}

Eigen::VectorXi nb2_invcdf_vec_blocks(const Eigen::VectorXd &alpha_vec,
                                      double m, double r) {
    double p = prob(m, r);
    return nb_invcdf_vec_blocks(alpha_vec, r, p);
}

Eigen::VectorXi zinb2_invcdf_vec_blocks_wrapper(
    const Eigen::VectorXd &alpha_vec, double m, double r, double pi) {
    double p = prob(m, r);

    Eigen::VectorXi nb_invcdf_results = nb_invcdf_vec_blocks(alpha_vec, r, p);
    Eigen::VectorXd alpha_adj_vec = (alpha_vec.array() - pi) / (1 - pi);

    Eigen::VectorXi zero_vec = Eigen::VectorXi::Zero(alpha_vec.size());

    Eigen::VectorXi zinb_invcdf_results =
        (alpha_adj_vec.array() <= 0).select(zero_vec, nb_invcdf_results);

    return zinb_invcdf_results;
}
