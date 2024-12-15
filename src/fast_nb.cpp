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

// Precompute log combinatorial coefficients for small values of k, r.
// This is only constexpr in GCC, but will be for both in C++26
#if defined(__GNUC__) && !defined(__clang__)
    #include "log_comb_gcc.hpp"
#else
    #include "log_comb.hpp"
#endif

// Threshold for switching to Stirling's approximation
const double STIRLING_THRESHOLD = 10.0;

double lgamma_stirling(double x) {
    if (x < STIRLING_THRESHOLD) {
        // For small values of x, use the standard lgamma function
        return std::lgamma(x);
    } else {
        // Stirling's approximation: ln(Gamma(x)) ≈ x*ln(x) - x + 0.5*ln(2*pi/x)
        static const double ln_sqrt_2pi = 0.9189385332046727; // ln(sqrt(2 * pi))
        return (x - 0.5) * std::log(x) - x + ln_sqrt_2pi;
    }
}

double compute_log_comb_sterling(int k, int r) {
    static const double lgamma_r = lgamma_stirling(r);  // Precompute lgamma(r)
    const double log_2pi = 0.9189385332046727;          // log(sqrt(2 * pi))

    if (k < STIRLING_THRESHOLD) {
        // Use exact lgamma for small k
        return lgamma_stirling(k + r) - lgamma_r - lgamma_stirling(k + 1);
    } else {
        // Use Stirling's approximation for large k
        double log_k = std::log(k);
        double log_k_r = std::log(k + r);

        return (k + r - 0.5) * log_k_r - (k + r)
             - (k + 0.5) * log_k + k
             - lgamma_r;
    }
}

double negative_binomial_pmf_expansion(int k, int r, double p) {
    if (k < 0) {
        return 0.0;
    }

    const double log_p = std::log(p);
    const double log_1_minus_p = std::log(1.0 - p);

    double log_comb = compute_log_comb_sterling(k, r);

    return std::exp(log_comb + k * log_1_minus_p + r * log_p);
}

double negative_binomial_pmf_stirling(int k, int r, double p) {
    if (k < 0) {
        return 0.0;
    }

    const double log_p = std::log(p);
    const double log_1_minus_p = std::log(1.0 - p);

    double log_comb = lgamma_stirling(k + r) - lgamma_stirling(r) - lgamma_stirling(k + 1);

    return std::exp(log_comb + k * log_1_minus_p + r * log_p);
}

// Fixed r version
double compute_log_comb(int k, int r, double lgamma_r) {
    double log_comb = 0.0;

    // Sum log(k + 1) to log(k + r - 1)
    for(int i = 1; i < r; ++i){
        log_comb += std::log(static_cast<double>(k) + static_cast<double>(i));
    }

    // Subtract precomputed lgamma(r)
    log_comb -= lgamma_r;

    return log_comb;
}

// Precompute lgamma(r) since r is constant
double precompute_lgamma_r(int r){
    return std::lgamma(static_cast<double>(r));
}

// Updated negative binomial PMF using optimized log_comb
double negative_binomial_pmf_optimized(int k, int r, double p, double lgamma_r) {
    if (k < 0) {
        return 0.0;
    }

    const double log_p = std::log(p);
    const double log_1_minus_p = std::log(1.0 - p);

    // Compute log_comb using the optimized function
    double log_comb = compute_log_comb(k, r, lgamma_r);

    // Compute log PMF and exponentiate
    return std::exp(log_comb + k * log_1_minus_p + r * log_p);
}

float negative_binomial_pmf_lut(int k, int r, float p) {

    // With k, r LUT (with GCC!)

    if (k < 0) {
        return 0.0;
    }

    const double log_p = std::log(p);
    const double log_1_minus_p = std::log(1.0 - p);

    // Use log-gamma for binomial coefficient: log(choose(k + r - 1, r - 1))
    double log_comb = compute_log_comb(k, r);

    // Compute log PMF and exponentiate
    return std::exp(log_comb + k * log_1_minus_p + r * log_p);
}

float negative_binomial_pmf_base(int k, int r, float p) {

    // The no optimisations version

    if (k < 0) {
        return 0.0;
    }

    const double log_p = std::log(p);
    const double log_1_minus_p = std::log(1.0 - p);

    // Use log-gamma for binomial coefficient: log(choose(k + r - 1, r - 1))
    double log_comb = std::lgamma(k + r) - std::lgamma(r) - std::lgamma(k + 1);

    // Compute log PMF and exponentiate
    return std::exp(log_comb + k * log_1_minus_p + r * log_p);
}
