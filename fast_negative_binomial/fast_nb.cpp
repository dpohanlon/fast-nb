#include <cmath>

#include "base_nb.hpp"
#include "eigen_nb.hpp"
#include "utils.hpp"
#include "vector_nb.hpp"

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

        return (k + r - 0.5) * log_k_r - (k + r) - (k + 0.5) * log_k + k -
               lgamma_r;
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

    double log_comb =
        lgamma_stirling(k + r) - lgamma_stirling(r) - lgamma_stirling(k + 1);

    return std::exp(log_comb + k * log_1_minus_p + r * log_p);
}
