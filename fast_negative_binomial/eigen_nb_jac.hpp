#pragma once

#include <omp.h>

#include <Eigen/Dense>
#include <boost/sort/sort.hpp>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <cmath>
#include <numeric>

#include "base_nb.hpp"
#include "eigen_nb.hpp"
#include "cache.hpp"
#include "utils.hpp"

#include <boost/math/special_functions/digamma.hpp>

// Precompute log combinatorial coefficients for small values of k, r.
// This is only constexpr in GCC, but will be for both in C++26
#if defined(__GNUC__) && !defined(__clang__)

#include "log_comb_gcc.hpp"
#else
#include "log_comb.hpp"
#endif

// Compute gradients block-wise for d(logNB)/dp and d(logNB)/dr.
//
//   d(logNB)/dp =  r/p  -  k/(1-p)
//   d(logNB)/dr =  1  +  digamma(r)  +  digamma(k + r)
//
// We reuse the previously computed gradient if k_block[i] repeats.
inline void compute_grad_block(const FixedVectorXi &k_block,
                               FixedVectorXd &grad_p_block,
                               FixedVectorXd &grad_r_block,
                               double r, double p,
                               double digamma_r) {
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        if (i > 0 && k_block[i] == k_block[i - 1]) {
            grad_p_block[i] = grad_p_block[i - 1];
            grad_r_block[i] = grad_r_block[i - 1];
        } else {
            double val_kr = boost::math::digamma(static_cast<double>(k_block[i]) + r);
            grad_p_block[i] = r / p - static_cast<double>(k_block[i]) / (1.0 - p);
            grad_r_block[i] = 1.0 + digamma_r + val_kr;
        }
    }
}

// Processes all complete blocks in parallel for the NB gradient.
void process_grad_blocks(const Eigen::VectorXi &k,
                         Eigen::VectorXd &grad_p,
                         Eigen::VectorXd &grad_r,
                         double r, double p,
                         int num_blocks) {

    double digamma_r = boost::math::digamma(r);

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int block = 0; block < num_blocks; ++block) {
            int start = block * BLOCK_SIZE;
            FixedVectorXi k_block = Eigen::Map<const FixedVectorXi>(k.data() + start);

            FixedVectorXd grad_p_block, grad_r_block;
            compute_grad_block(k_block, grad_p_block, grad_r_block, r, p, digamma_r);

            Eigen::Map<FixedVectorXd>(grad_p.data() + start) = grad_p_block;
            Eigen::Map<FixedVectorXd>(grad_r.data() + start) = grad_r_block;
        }
    }
}

Eigen::MatrixXd process_grad_remaining(const Eigen::VectorXi &k,
                                       int start,
                                       int remaining,
                                       double r,
                                       double p) {
    Eigen::MatrixXd out(remaining, 2);

    double digamma_r = boost::math::digamma(r);
    for (int i = 0; i < remaining; ++i) {
        if (i > 0 && k[start + i] == k[start + i - 1]) {
            out(i, 0) = out(i - 1, 0);
            out(i, 1) = out(i - 1, 1);
        } else {
            double val_kr = boost::math::digamma(static_cast<double>(k[start + i]) + r);
            out(i, 0) = r / p - static_cast<double>(k[start + i]) / (1.0 - p);
            out(i, 1) = 1.0 + digamma_r + val_kr;
        }
    }

    return out;
}

template <typename T>
Eigen::MatrixXd nb_gradient_vec_eigen_blocks_no_copy(Eigen::Ref<Eigen::VectorXi> k,
                                                     T r,
                                                     double p) {
    // Sort k in-place
    boost::sort::parallel_stable_sort(k.data(), k.data() + k.size());

    // Prepare storage
    Eigen::MatrixXd grad(k.size(), 2);
    Eigen::VectorXd grad_p(k.size()), grad_r(k.size());

    const int num_blocks = k.size() / BLOCK_SIZE;
    const int remaining = k.size() % BLOCK_SIZE;

    // Process complete blocks in parallel
    if (num_blocks > 0) {
        process_grad_blocks(k, grad_p, grad_r, r, p, num_blocks);
    }

    // Process the remainder
    if (remaining > 0) {
        int start = num_blocks * BLOCK_SIZE;
        Eigen::MatrixXd tail = process_grad_remaining(k, start, remaining, r, p);
        grad_p.segment(start, remaining) = tail.col(0);
        grad_r.segment(start, remaining) = tail.col(1);
    }

    // Combine into a single MatrixXd
    grad.col(0) = grad_p;
    grad.col(1) = grad_r;

    return grad;
}

template <typename T>
Eigen::MatrixXd nb_gradient_vec_eigen_blocks(const Eigen::VectorXi &k_in,
                                             T r,
                                             double p) {
    Eigen::VectorXi k_copy = k_in;
    return nb_gradient_vec_eigen_blocks_no_copy(k_copy, r, p);
}

Eigen::VectorXd nb2_gradient__vec_eigen_blocks(const Eigen::VectorXi &k, double m,
                                          double r) {
    // m : mean
    // r : concentration

    double p = prob(m, r);

    return nb_gradient_vec_eigen_blocks(k, r, p);
}

Eigen::VectorXd nb2_gradient_vec_eigen_blocks_no_copy(Eigen::Ref<Eigen::VectorXi> k, double m,
                                                  double r) {
    // m : mean
    // r : concentration

    double p = prob(m, r);

    return nb_gradient_vec_eigen_blocks_no_copy(k, r, p);
}

template <typename T>
Eigen::MatrixXd zinb_gradient_vec_eigen_blocks_post_process_select(const Eigen::VectorXi &k_in,
                                                                   double m,
                                                                   double r,
                                                                   double alpha) {

    double p = prob(m, r);

    // Probably quite slow to do it this way
    Eigen::MatrixXd nb_grad = nb_gradient_vec_eigen_blocks(k_in, r, p);
    Eigen::VectorXd nb_probs = nb_base_vec_eigen_blocks(k_in, r, p);

    Eigen::MatrixXd zinb_grad(k_in.size(), 3);
    zinb_grad.leftCols(2) = nb_grad;

    Eigen::VectorXd alpha_grad_if_zero = (1 - nb_probs.array()) / (alpha + (1 - alpha) * nb_probs.array());

    Eigen::VectorXd alpha_grad_if_nonzero = -nb_probs.array() / ((1 - alpha) * nb_probs.array());

    zinb_grad.col(2) = (k_in.array() == 0).select(alpha_grad_if_zero, alpha_grad_if_nonzero);

    return zinb_grad;
}
