#include <vector>
#include <cmath>
#include <numeric>
#include <benchmark/benchmark.h>

#include <iostream>
#include <vector>
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

// Function to test accuracy and collect divergences
void test_accuracy(double tolerance, int max_k, const std::vector<int>& r_values, const std::vector<double>& p_values) {
    std::cout << "Testing accuracy with tolerance: " << tolerance << "\n";

    // Map to store divergences: key as k, value as vector of tuples (r, p, difference)
    std::map<int, std::vector<std::tuple<int, double, double>>> divergences;

    for (int r : r_values) {
        double lgamma_r = precompute_lgamma_r(r);
        for (double p : p_values) {
            for (int k = 0; k <= max_k; ++k) {
                double base_value = negative_binomial_pmf_base(k, r, p);
                double optimized_value = negative_binomial_pmf_optimized(k, r, p, lgamma_r);
                double diff = std::abs(base_value - optimized_value);

                if (diff > tolerance) {
                    divergences[k].emplace_back(r, p, diff);
                }
            }
        }
    }

    // Print results
    if (divergences.empty()) {
        std::cout << "No divergences found within the specified tolerance.\n";
    } else {
        std::cout << std::setw(10) << "k" << std::setw(10) << "r" << std::setw(12) << "p"
                  << std::setw(15) << "Difference" << "\n";

        for (const auto& [k, entries] : divergences) {
            for (const auto& [r, p, diff] : entries) {
                std::cout << std::setw(10) << k << std::setw(10) << r << std::setw(12) << p
                          << std::setw(15) << diff << "\n";
            }
        }

        std::cout << std::endl;

        std::cout << "Top 10 divergences:\n";

        std::cout << std::setw(10) << "k"
                  << std::setw(10) << "r"
                  << std::setw(12) << "p"
                  << std::setw(15) << "Difference" << "\n";

        // Temporary vector to hold all divergences (k, r, p, diff)
        std::vector<std::tuple<int, int, double, double>> flat_divergences;

        // Flatten the map into a single vector
        for (const auto& [k, entries] : divergences) {
            for (const auto& [r, p, diff] : entries) {
                flat_divergences.emplace_back(k, r, p, diff);
            }
        }

        // Sort the flattened vector by the difference in descending order
        std::sort(flat_divergences.begin(), flat_divergences.end(),
                  [](const auto& a, const auto& b) { return std::get<3>(a) > std::get<3>(b); });

        size_t limit = 10;

        // Print the top 10 divergences
        for (size_t i = 0; i < std::min(limit, flat_divergences.size()); ++i) {
            const auto& [k, r, p, diff] = flat_divergences[i];
            std::cout << std::setw(10) << k
                      << std::setw(10) << r
                      << std::setw(12) << p
                      << std::setw(15) << diff << "\n";
        }

    }
}

// int main() {
//     // Tolerance for comparison
//     double tolerance = 1e-6;

//     // Maximum value of k (number of failures)
//     int max_k = 100;

//     // Set of r values (number of successes)
//     std::vector<int> r_values = {1, 5, 10, 20, 50, 100, 200, 500, 1000};

//     // Set of p values (success probabilities)
//     std::vector<double> p_values = {0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99};

//     // Run the accuracy test
//     test_accuracy(tolerance, max_k, r_values, p_values);

//     return 0;
// }

static void BM_NegativeBinomialPMF(benchmark::State& state) {
    int r = 5;
    float p = 0.7;
    std::vector<int> k_vals(state.range(0));
    std::iota(k_vals.begin(), k_vals.end(), 0);

    // float lgamma_r = precompute_lgamma_r(r);

    for (auto _ : state) {
        std::vector<float> results(k_vals.size());
        for (size_t i = 0; i < k_vals.size(); ++i) {
            // results[i] = negative_binomial_pmf_base(k_vals[i], r, p);
            results[i] = negative_binomial_pmf_lut(k_vals[i], r, p);
            // results[i] = negative_binomial_pmf_optimized(k_vals[i], r, p, lgamma_r);
        }
        benchmark::DoNotOptimize(results);
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_NegativeBinomialPMF)->Range(16, 1 << 20)->Complexity();

BENCHMARK_MAIN();
