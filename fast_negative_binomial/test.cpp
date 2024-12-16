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
                // double optimized_value = negative_binomial_pmf_optimized(k, r, p, lgamma_r);
                double optimized_value = negative_binomial_pmf_stirling(k, r, p);
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
