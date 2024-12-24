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

#include <omp.h>

// Precompute log combinatorial coefficients for small values of k, r.
// This is only constexpr in GCC, but will be for both in C++26
#if defined(__GNUC__) && !defined(__clang__)
    // #include <omp.h>
    #include "log_comb_gcc.hpp"
#else
    #include "log_comb.hpp"
#endif

// Is this actually doing anything?

class LgammaCachedSorted {
public:
    LgammaCachedSorted() : current_k_(0), current_lgamma_(0.0), relative_cost_(10) {}

    // Compute lgamma(x) with caching for sorted x
    double lgamma(int x) {

        // std::cout << "current_k " << current_k_ << std::endl;
        // std::cout << "current_lgamma_ " << current_lgamma_ << std::endl;
        // std::cout << std::endl;

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

class LgammaCacheMap {
public:
    LgammaCacheMap() : max_k_(0), relative_cost_(10) {}

    double lgamma(int x) {

        if (x < 1) {
            std::cerr << "Error: x must be positive integer." << std::endl;
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (x == max_k_) {
            return lgamma_cache_[x];
        } else if (x > max_k_ + relative_cost_) {
            double lg = std::lgamma(static_cast<double>(x));

            max_k_ = x;
            lgamma_cache_[x] = lg;

            return lg;

        } else {

            // Compute iteratively from current_k_ to x
            while (max_k_ < x) {
                if (max_k_ == 0) {
                    // Initialize lgamma(1) = 0
                    max_k_ = 1;
                    lgamma_cache_[1] = 0.0;
                } else {
                    // Use the identity: lgamma(x + 1) = log(x) + lgamma(x)
                    lgamma_cache_[max_k_ + 1] = lgamma_cache_[max_k_] + std::log(static_cast<double>(max_k_));
                    max_k_++;
                }
            }

            return lgamma_cache_[x];

        }
    }

private:
    std::unordered_map<int, double> lgamma_cache_;
    int max_k_;
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

    // std::cout << "lg_kr" << std::endl;
    double lg_kr = lgamma_kr.lgamma(k + r);

    // std::cout << "lg_k1" << std::endl;
    double lg_k1 = lgamma_k1.lgamma(k + 1);

    double log_comb = lg_kr - lgamma_r - lg_k1;

    return std::exp(log_comb + k * log_1_minus_p + r * log_p);
}

double nb_base_fixed_r_opt(int k, int r, double p, double lgamma_r, LgammaCacheMap & lgamma_cache) {

    if (k < 0) {
        return 0.0;
    }

    const double log_p = std::log(p);
    const double log_1_minus_p = std::log(1.0 - p);

    double lg_kr = lgamma_cache.lgamma(k + r);
    double lg_k1 = lgamma_cache.lgamma(k + 1);

    double log_comb = lg_kr - lgamma_r - lg_k1;

    return std::exp(log_comb + k * log_1_minus_p + r * log_p);
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
Eigen::VectorXd nb_base_vec_eigen(Eigen::VectorXi &k, T r, double p)
{
    double lgamma_r = std::lgamma(static_cast<double>(r));
    Eigen::VectorXd results(k.size());

    boost::sort::spreadsort::spreadsort(k.begin(), k.end());

    int k_prev = -1;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < k.size(); ++i) {

        if (k[i] == k_prev) {
            results[i] = results[i - 1];
        } else {
            results[i] = nb_base_fixed_r(k[i], r, p, lgamma_r);
        }

        k_prev = k[i];
    }

    return results;
}

// Maybe having a cache rather than sorting is a better idea anyway? It can persist between calls with same k, different r, p

template<typename T>
Eigen::VectorXd nb_base_vec_eigen_sorted(Eigen::VectorXi &k, T r, double p)
{
    double lgamma_r = std::lgamma(static_cast<double>(r));
    Eigen::VectorXd results(k.size());

    boost::sort::spreadsort::spreadsort(k.begin(), k.end()); // Faster for smaller data

    LgammaCachedSorted lgamma_kr;
    LgammaCachedSorted lgamma_k1;

    int k_prev = -1;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < k.size(); ++i) {
        if (k[i] == k_prev) {
            results[i] = results[i - 1];
        } else {
            results[i] = nb_base_fixed_r_opt(k[i], r, p, lgamma_r, lgamma_kr, lgamma_k1);
        }
        k_prev = k[i];
    }

    return results;
}

// Define the fixed block size (tune this based on your needs)
constexpr int BLOCK_SIZE = 2048;

// Define fixed-size Eigen vector types for integers and doubles
using FixedVectorXi = Eigen::Matrix<int, BLOCK_SIZE, 1>;
using FixedVectorXd = Eigen::Matrix<double, BLOCK_SIZE, 1>;

// Template function to compute the negative binomial PMF using Eigen
template<typename T>
Eigen::VectorXd nb_base_vec_eigen_blocks(Eigen::VectorXi &k, T r, double p) {

    const double lgamma_r = std::lgamma(static_cast<double>(r));

    boost::sort::parallel_stable_sort(k.begin(), k.end());

    Eigen::VectorXd results(k.size());

    const int num_blocks = static_cast<int>(k.size()) / BLOCK_SIZE;
    const int remaining = static_cast<int>(k.size()) % BLOCK_SIZE;

    const double log_p = std::log(p);
    const double log_1_minus_p = std::log(1.0 - p);

    #pragma omp parallel for schedule(static)
    for(int block = 0; block < num_blocks; ++block) {
        const int start = block * BLOCK_SIZE;

        Eigen::Map<const FixedVectorXi> k_block(k.data() + start);

        FixedVectorXd res_block;

        for(int i = 0; i < BLOCK_SIZE; ++i) {

            if ((i > 0) && (k[i - 1] == k[i])) {
                res_block[i] = res_block[i - 1];
                continue;
            } else {

                double log_comb = std::lgamma(k_block[i] + r) - lgamma_r - std::lgamma(k_block[i] + 1);

                res_block[i] = std::exp(log_comb + k_block[i] * log_1_minus_p + r * log_p);

                // res_block[i] = nb_base_fixed_r(k_block[i], r, p, lgamma_r);

            }
        }

        Eigen::Map<FixedVectorXd>(results.data() + start) = res_block;
    }

    if(remaining > 0) {

        const int start = num_blocks * BLOCK_SIZE;

        Eigen::VectorXi k_remaining = k.segment(start, remaining);

        Eigen::VectorXd res_remaining(remaining);

        for(int i = 0; i < remaining; ++i) {

            if ((i > 0) && (k_remaining[i - 1] == k_remaining[i])) {
                res_remaining[i] = res_remaining[i - 1];
                continue;
            } else {

                res_remaining[i] = nb_base_fixed_r(k_remaining[i], r, p, lgamma_r);
            }
        }

        results.segment(start, remaining) = res_remaining;
    }

    return results;
}
