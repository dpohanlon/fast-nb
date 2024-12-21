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

#include <boost/sort/sort.hpp>
#include <boost/sort/spreadsort/spreadsort.hpp>

#include <utils.hpp>

// Precompute log combinatorial coefficients for small values of k, r.
// This is only constexpr in GCC, but will be for both in C++26
#if defined(__GNUC__) && !defined(__clang__)
    #include <omp.h>
    #include "log_comb_gcc.hpp"
#else
    #include "log_comb.hpp"
#endif

class LgammaCachedSorted {
public:
    LgammaCachedSorted() : current_k_(0), current_lgamma_(0.0), relative_cost_(20) {}

    // Compute lgamma(x) with caching for sorted x
    double lgamma(int x) {
        if (x < 1) {
            std::cerr << "Error: x must be positive integer." << std::endl;
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (x == current_k_) {
            return current_lgamma_;
        } else if (x > current_k_ + relative_cost_) {
            current_lgamma_ = std::lgamma(static_cast<double>(x));
            current_k_ = x;

            return current_lgamma_;

        } else {

            // Compute iteratively from current_k_ to x
            while (current_k_ < x) {
                if (current_k_ == 0) {
                    // Initialize lgamma(1) = 0
                    current_k_ = 1;
                    current_lgamma_ = 0.0;
                } else {
                    // Use the identity: lgamma(x + 1) = log(x) + lgamma(x)
                    current_lgamma_ += std::log(static_cast<double>(current_k_));
                    current_k_++;
                }
            }

            return current_lgamma_;

        }
    }

private:
    int current_k_;        // The current largest k computed
    double current_lgamma_; // The current lgamma(k) value
    int relative_cost_;
};

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

double nb_base_fixed_r_opt(int k, int r, double p, double lgamma_r, LgammaCachedSorted & lgamma_kr, LgammaCachedSorted & lgamma_k1) {

    if (k < 0) {
        return 0.0;
    }

    const double log_p = std::log(p);
    const double log_1_minus_p = std::log(1.0 - p);

    double log_comb = lgamma_kr.lgamma(k + r) - lgamma_r - lgamma_k1.lgamma(k + 1);

    // return std::exp(log_comb + k * log_1_minus_p + r * log_p);
    return log_comb + k * log_1_minus_p + r * log_p;
}

// I can probably template specialise these for vector or scalar, but not sure if it's worth it

template<typename T>
std::vector<double> nb_base_vec(std::vector<int> k, T r, double p)
{
    double lgamma_r = std::lgamma(static_cast<double>(r));

    // std::sort(k.begin(), k.end());
    // boost::sort::parallel_stable_sort(k.begin(), k.end());

    std::vector<double> results(k.size());

    for (int i = 0; i < k.size(); ++i) {
        results[i] = nb_base_fixed_r(k[i], r, p, lgamma_r);
    }

    // Even though this has a second loop and more memory accesses, it's SIMD and cache (?) friendlier than doing the exp within the call (exp is slow and these values are often the same)

    for (int i = 0; i < k.size(); ++i) {
        results[i] = std::exp(results[i]);
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

template<typename T>
Eigen::VectorXd nb_base_vec_eigen_sorted(Eigen::VectorXi &k, T r, double p)
{
    double lgamma_r = std::lgamma(static_cast<double>(r));
    Eigen::VectorXd results(k.size());

    // std::sort(k.begin(), k.end());
    // boost::sort::parallel_stable_sort(k.begin(), k.end());
    boost::sort::spreadsort::spreadsort(k.begin(), k.end());

    LgammaCachedSorted lgamma_kr;
    LgammaCachedSorted lgamma_k1;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < k.size(); ++i) {
        results[i] = nb_base_fixed_r_opt(k[i], r, p, lgamma_r, lgamma_kr, lgamma_k1);
    }

    return results;
}

// Define the fixed block size (tune this based on your needs)
constexpr int BLOCK_SIZE = 1024;

// Define fixed-size Eigen vector types for integers and doubles
using FixedVectorXi = Eigen::Matrix<int, BLOCK_SIZE, 1>;
using FixedVectorXd = Eigen::Matrix<double, BLOCK_SIZE, 1>;

// Template function to compute the negative binomial PMF using Eigen
template<typename T>
Eigen::VectorXd nb_base_vec_eigen_blocks(const Eigen::VectorXi &k, T r, double p) {
    // Precompute lgamma(r) since it's constant across all computations
    const double lgamma_r = std::lgamma(static_cast<double>(r));

    // Initialize the result vector with the same size as k
    Eigen::VectorXd results(k.size());

    // Calculate the number of full blocks and the number of remaining elements
    const int num_blocks = static_cast<int>(k.size()) / BLOCK_SIZE;
    const int remaining = static_cast<int>(k.size()) % BLOCK_SIZE;

    // Parallelize the processing of full blocks using OpenMP
    #pragma omp parallel for schedule(static)
    for(int block = 0; block < num_blocks; ++block) {
        // Calculate the starting index for the current block
        const int start = block * BLOCK_SIZE;

        // Map the current block of k to a fixed-size Eigen vector
        // Ensure that k has enough elements to map; this is safe since we're iterating over full blocks
        Eigen::Map<const FixedVectorXi> k_block(k.data() + start);

        // Initialize a fixed-size Eigen vector to store the results of the current block
        FixedVectorXd res_block;

        // Compute nb_base_fixed_r for each element in the block
        for(int i = 0; i < BLOCK_SIZE; ++i) {

            const double log_p = std::log(p);
            const double log_1_minus_p = std::log(1.0 - p);

            double log_comb = std::lgamma(k_block[i] + r) - lgamma_r - std::lgamma(k_block[i] + 1);

            res_block[i] = std::exp(log_comb + k_block[i] * log_1_minus_p + r * log_p);

            // res_block[i] = nb_base_fixed_r(k_block[i], r, p, lgamma_r);
        }

        // Assign the computed results back to the corresponding segment in the results vector
        Eigen::Map<FixedVectorXd>(results.data() + start) = res_block;
    }

    // Handle any remaining elements that don't fit into a full block
    if(remaining > 0) {
        // Calculate the starting index for the remaining elements
        const int start = num_blocks * BLOCK_SIZE;

        // Extract the remaining segment from k
        Eigen::VectorXi k_remaining = k.segment(start, remaining);

        // Initialize a dynamic-size Eigen vector to store the results of the remaining elements
        Eigen::VectorXd res_remaining(remaining);

        // Compute nb_base_fixed_r for each remaining element
        for(int i = 0; i < remaining; ++i) {
            res_remaining[i] = nb_base_fixed_r(k_remaining[i], r, p, lgamma_r);
        }

        // Assign the computed results back to the corresponding segment in the results vector
        results.segment(start, remaining) = res_remaining;
    }

    return results;
}
