#pragma once

#include <cmath>
#include <numeric>
#include <vector>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
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

class LgammaCache {
public:
  LgammaCache() : current_k_(0), current_lgamma_(0.0), relative_cost_(10) {}

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
  int current_k_;         // The current largest k computed
  double current_lgamma_; // The current lgamma(k) value
  int relative_cost_;
};

// The 'lossless' optimisations versions

template <typename T> double nb_base(int k, T r, double p) {

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

template <> double nb_base_fixed_r(int k, int r, double p, double lgamma_r) {

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

double nb_base_fixed_r_opt(int k, int r, double p, double lgamma_r,
                           LgammaCache &lgamma_kr, LgammaCache &lgamma_k1) {

  if (k < 0) {
    return 0.0;
  }

  const double log_p = std::log(p);
  const double log_1_minus_p = std::log(1.0 - p);

  double lg_kr = lgamma_kr.lgamma(k + r);
  double lg_k1 = lgamma_k1.lgamma(k + 1);

  double log_comb = lg_kr - lgamma_r - lg_k1;

  return std::exp(log_comb + k * log_1_minus_p + r * log_p);
}

// I can probably template specialise these for vector or scalar, but not sure
// if it's worth it

template <typename T>
std::vector<double> nb_base_vec(std::vector<int> k, T r, double p) {
  double lgamma_r = std::lgamma(static_cast<double>(r));

  std::vector<double> results(k.size());

  for (int i = 0; i < k.size(); ++i) {
    results[i] = nb_base_fixed_r(k[i], r, p, lgamma_r);
  }

  return results;
}

template <typename T>
Eigen::VectorXd nb_base_vec_eigen(const Eigen::VectorXi &k_in, T r, double p) {
  Eigen::VectorXi k = k_in;

  double lgamma_r = std::lgamma(static_cast<double>(r));
  Eigen::VectorXd results(k.size());

  boost::sort::spreadsort::spreadsort(k.begin(),
                                      k.end()); // Faster for smaller data

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
                                         double p) {
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
      results[i] =
          nb_base_fixed_r_opt(k[i], r, p, lgamma_r, lgamma_kr, lgamma_k1);
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

/**
 * @brief Computes the Negative Binomial PMF for a single block.
 *
 * @param k_block           The block of k values.
 * @param res_block         Reference to store the computed PMF results.
 * @param lgamma_r          Precomputed log gamma of r.
 * @param log_p             Precomputed log(p).
 * @param log_1_minus_p     Precomputed log(1 - p).
 * @param r                 The 'r' parameter of the Negative Binomial
 * distribution.
 */
inline void compute_pmf_block(const FixedVectorXi &k_block,
                              FixedVectorXd &res_block, const double lgamma_r,
                              const double log_p, const double log_1_minus_p,
                              const double r) {
  for (int i = 0; i < BLOCK_SIZE; ++i) {
    if ((i > 0) && (k_block[i - 1] == k_block[i])) {
      res_block[i] = res_block[i - 1];
    } else {
      const double log_comb =
          std::lgamma(static_cast<double>(k_block[i]) + r) - lgamma_r -
          std::lgamma(static_cast<double>(k_block[i]) + 1.0);
      res_block[i] =
          std::exp(log_comb + static_cast<double>(k_block[i]) * log_1_minus_p +
                   r * log_p);
    }
  }
}

/**
 * @brief Processes all complete blocks in parallel using OpenMP.
 *
 * @param k                 The sorted vector of k values.
 * @param results           Reference to store all computed PMF results.
 * @param lgamma_r          Precomputed log gamma of r.
 * @param log_p             Precomputed log(p).
 * @param log_1_minus_p     Precomputed log(1 - p).
 * @param r                 The 'r' parameter of the Negative Binomial
 * distribution.
 * @param num_blocks        The number of complete blocks to process.
 */
void process_blocks(const Eigen::VectorXi &k, Eigen::VectorXd &results,
                    const double lgamma_r, const double log_p,
                    const double log_1_minus_p, const double r,
                    const int num_blocks) {

#pragma omp parallel
  {

#pragma omp for schedule(static)
    for (int block = 0; block < num_blocks; ++block) {
      const int start = block * BLOCK_SIZE;
      const FixedVectorXi k_block =
          Eigen::Map<const FixedVectorXi>(k.data() + start);

      FixedVectorXd res_block;
      compute_pmf_block(k_block, res_block, lgamma_r, log_p, log_1_minus_p, r);

      Eigen::Map<FixedVectorXd>(results.data() + start) = res_block;
    }
  }
}

/**
 * @brief Processes the remaining elements that do not fit into a complete
 * block.
 *
 * @param k         The sorted vector of k values.
 * @param start     The starting index of the remaining segment.
 * @param remaining The number of remaining elements.
 * @param r         The 'r' parameter of the Negative Binomial distribution.
 * @param p         The probability parameter of the Negative Binomial
 * distribution.
 * @return Eigen::VectorXd The computed PMF for the remaining elements.
 */
Eigen::VectorXd process_remaining(const Eigen::VectorXi &k, const int start,
                                  const int remaining, const double r,
                                  const double p) {
  const Eigen::VectorXi k_remaining = k.segment(start, remaining);
  return nb_base_vec_eigen_sorted(k_remaining, r, p);
}

/**
 * @brief Computes the Negative Binomial PMF in blocks, leveraging parallel
 * processing and Eigen optimizations.
 *
 * @tparam T Type of the 'r' parameter (e.g., int, double).
 * @param k     The vector of k values (will be sorted in-place).
 * @param r     The 'r' parameter of the Negative Binomial distribution.
 * @param p     The probability parameter of the Negative Binomial distribution.
 * @return Eigen::VectorXd The computed PMF values.
 */
template <typename T>
Eigen::VectorXd nb_base_vec_eigen_blocks_no_copy(Eigen::VectorXi &k, T r,
                                                 double p) {
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
    process_blocks(k, results, lgamma_r, log_p, log_1_minus_p, r_d, num_blocks);
  }

  // Process any remaining elements
  if (remaining > 0) {
    const int start = num_blocks * BLOCK_SIZE;
    results.segment(start, remaining) =
        process_remaining(k, start, remaining, r_d, p);
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

double nb2_base(int k, double m, double r) {
  // m : mean
  // r : concentration

  double p = prob(m, r);

  return nb_base(k, r, p);
}

std::vector<double> nb2_base_vec(std::vector<int> k, double m, double r) {
  // m : mean
  // r : concentration

  double p = prob(m, r);

  return nb_base_vec(k, r, p);
}

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

Eigen::VectorXd nb2_base_vec_eigen_blocks_no_copy(Eigen::VectorXi &k, double m,
                                          double r) {
  // m : mean
  // r : concentration

  double p = prob(m, r);

  return nb_base_vec_eigen_blocks_no_copy(k, r, p);
}
