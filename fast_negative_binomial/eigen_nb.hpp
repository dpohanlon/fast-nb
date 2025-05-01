#pragma once

#include <omp.h>

#include <Eigen/Dense>
#include <boost/sort/sort.hpp>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <cmath>
#include <numeric>

#include "base_nb.hpp"
#include "cache.hpp"
#include "utils.hpp"

// Precompute log combinatorial coefficients for small values of k, r.
// This is only constexpr in GCC, but will be for both in C++26
#if defined(__GNUC__) && !defined(__clang__)

#include "log_comb_gcc.hpp"
#else
#include "log_comb.hpp"
#endif

template <typename T>
Eigen::VectorXd nb_base_vec_eigen(const Eigen::VectorXi &k_in, T r, double p) {
    Eigen::VectorXi k = k_in;

    double lgamma_r = std::lgamma(static_cast<double>(r));
    Eigen::VectorXd results(k.size());

    boost::sort::spreadsort::spreadsort(k.begin(),
                                        k.end());

    LgammaCache lgamma_kr;
    LgammaCache lgamma_k1;

    int k_prev = -1;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < k.size(); ++i) {
        if (k[i] == k_prev) {
            results[i] = results[i - 1];
        } else {
            results[i] =
                nb_base_fixed_r_opt(k[i], r, p, lgamma_r, lgamma_kr, lgamma_k1);
        }
        k_prev = k[i];
    }

    return results;
}

// Assumed sorted (-> no copy)
template <typename T>
Eigen::VectorXd nb_base_vec_eigen_sorted(const Eigen::VectorXi &k, T r,
                                         double p, bool log = false) {
    double lgamma_r = std::lgamma(static_cast<double>(r));
    Eigen::VectorXd results(k.size());

    LgammaCache lgamma_kr;
    LgammaCache lgamma_k1;

    int k_prev = -1;

    // Not worth a parallel for here for small data size
    for (int i = 0; i < k.size(); ++i) {
        if (k[i] == k_prev) {
            results[i] = results[i - 1];
        } else {
            results[i] = nb_base_fixed_r_opt(k[i], r, p, lgamma_r, lgamma_kr,
                                             lgamma_k1, log);
        }
        k_prev = k[i];
    }

    return results;
}

inline void compute_pmf_block(const FixedVectorXi &k_block,
                              FixedVectorXd &res_block, const double lgamma_r,
                              const double log_p, const double log_1_minus_p,
                              const double r, LgammaCache &lgamma_kr,
                              LgammaCache &lgamma_k1, bool log = false) {
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        if ((i > 0) && (k_block[i - 1] == k_block[i])) {
            res_block[i] = res_block[i - 1];
        } else {
            const double log_comb = lgamma_kr.lgamma(k_block[i] + r) -
                                    lgamma_r - lgamma_k1.lgamma(k_block[i] + 1);
            double res = log_comb +
                         static_cast<double>(k_block[i]) * log_1_minus_p +
                         r * log_p;
            res_block[i] = log ? res : std::exp(res);
        }
    }
}

void process_blocks(const Eigen::VectorXi &k, Eigen::VectorXd &results,
                    const double lgamma_r, const double log_p,
                    const double log_1_minus_p, const double r,
                    const int num_blocks, bool log = false) {
#pragma omp parallel
    {
        // Each thread should have its own cache
        LgammaCache lgamma_kr;
        LgammaCache lgamma_k1;

#pragma omp for schedule(static)
        for (int block = 0; block < num_blocks; ++block) {
            const int start = block * BLOCK_SIZE;
            const FixedVectorXi k_block =
                Eigen::Map<const FixedVectorXi>(k.data() + start);

            FixedVectorXd res_block;
            compute_pmf_block(k_block, res_block, lgamma_r, log_p,
                              log_1_minus_p, r, lgamma_kr, lgamma_k1, log);

            Eigen::Map<FixedVectorXd>(results.data() + start) = res_block;
        }
    }
}

Eigen::VectorXd process_remaining(const Eigen::VectorXi &k, const int start,
                                  const int remaining, const double r,
                                  const double p, bool log = false) {
    const Eigen::VectorXi k_remaining = k.segment(start, remaining);
    return nb_base_vec_eigen_sorted(k_remaining, r, p, log);
}

template <typename T>
Eigen::VectorXd nb_base_vec_eigen_blocks_no_copy(Eigen::Ref<Eigen::VectorXi> k,
                                                 T r, double p,
                                                 bool log = false) {
    // Precompute constants
    const double r_d = static_cast<double>(r);
    const double lgamma_r = std::lgamma(r_d);
    const double log_p = std::log(p);
    const double log_1_minus_p = std::log(1.0 - p);

    // Sort k in-place
    boost::sort::parallel_stable_sort(k.data(), k.data() + k.size());

    // Initialize results vector
    Eigen::VectorXd results(k.size());

    // Determine the number of complete blocks and remaining elements
    const int num_blocks = k.size() / BLOCK_SIZE;
    const int remaining = k.size() % BLOCK_SIZE;

    // Process all complete blocks
    if (num_blocks > 0) {
        process_blocks(k, results, lgamma_r, log_p, log_1_minus_p, r_d,
                       num_blocks, log);
    }

    // Process any remaining elements
    if (remaining > 0) {
        const int start = num_blocks * BLOCK_SIZE;
        results.segment(start, remaining) =
            process_remaining(k, start, remaining, r_d, p, log);
    }

    return results;
}

template <typename T>
Eigen::VectorXd nb_base_vec_eigen_blocks(const Eigen::VectorXi &k_in, T r,
                                         double p) {
    // Copy to avoid modifying the input vector - there is some overhead here
    Eigen::VectorXi k = k_in;
    return nb_base_vec_eigen_blocks_no_copy(k, r, p);
}

// Wrappers for nb2

Eigen::VectorXd nb2_base_vec_eigen(const Eigen::VectorXi &k, double m,
                                   double r) {
    // m : mean
    // r : concentration

    double p = prob(m, r);

    return nb_base_vec_eigen(k, r, p);
}

Eigen::VectorXd nb2_base_vec_eigen_blocks(const Eigen::VectorXi &k, double m,
                                          double r) {
    // m : mean
    // r : concentration

    double p = prob(m, r);

    return nb_base_vec_eigen_blocks(k, r, p);
}

Eigen::VectorXd nb2_base_vec_eigen_blocks_no_copy(Eigen::Ref<Eigen::VectorXi> k,
                                                  double m, double r) {
    // m : mean
    // r : concentration

    double p = prob(m, r);

    return nb_base_vec_eigen_blocks_no_copy(k, r, p);
}

Eigen::VectorXd log_nb2_base_vec_eigen_blocks_no_copy(
    Eigen::Ref<Eigen::VectorXi> k, double m, double r) {
    // m : mean
    // r : concentration

    double p = prob(m, r);

    return nb_base_vec_eigen_blocks_no_copy(k, r, p, true);
}

Eigen::VectorXd zinb2_base_vec_eigen_blocks(const Eigen::VectorXi &k, double m,
                                            double r, double alpha) {
    // m : mean
    // r : concentration

    double p = prob(m, r);

    Eigen::VectorXd probs = nb_base_vec_eigen_blocks(k, r, p);

    Eigen::VectorXd scaled_probs = probs * (1.0 - alpha);

    Eigen::VectorXd zero_vec = Eigen::VectorXd::Zero(k.size());
    Eigen::VectorXd alpha_vec = Eigen::VectorXd::Constant(k.size(), alpha);

    Eigen::VectorXd zero_inflation =
        (k.array() == 0).select(alpha_vec, zero_vec);
    Eigen::VectorXd zinb_probs = scaled_probs.array() + zero_inflation.array();

    return zinb_probs;
}
