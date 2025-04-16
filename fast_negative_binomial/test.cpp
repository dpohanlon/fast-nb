#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <tuple>
#include <vector>

#include "optimise.hpp"

// Function to generate a matrix of counts from a truncated normal distribution.
Eigen::MatrixXi generateTruncatedNormalMatrix(int rows, int cols, double mean, double std_dev) {
    Eigen::MatrixXi mat(rows, cols);
    // Using std::random_device to seed the generator.
    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<double> dist(mean, std_dev);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double sample = dist(rng);
            // Truncate at 0.
            int count = sample < 0.0 ? 0 : static_cast<int>(sample);
            mat(i, j) = count;
        }
    }
    return mat;
}

// void test_optimisation() {
//     Eigen::MatrixXi matrix = generateTruncatedNormalMatrix(150, 10000, 50, 3);

//     std::vector<double> m_vec(150, 50.);
//     std::vector<double> r_vec(150, 10.0);

//     try {
//         // Call the optimisation function for all genes.
//         auto optimized_params = optimise_all_genes(matrix, m_vec, r_vec);

//         // Print the optimized parameters.
//         for (size_t i = 0; i < optimized_params.size(); ++i) {
//             std::cout << "Gene " << i
//                       << ": Optimised m = " << optimized_params[i].first
//                       << ", r = " << optimized_params[i].second << std::endl;
//         }
//     } catch (const std::exception& ex) {
//         std::cerr << "Exception: " << ex.what() << std::endl;
//     }
// }

// Function to test accuracy and collect divergences
void test_accuracy(double tolerance, int max_k,
                   const std::vector<int> &r_values,
                   const std::vector<double> &p_values) {
    std::cout << "Testing accuracy with tolerance: " << tolerance << "\n";

    // Map to store divergences: key as k, value as vector of tuples (r, p,
    // difference)
    std::map<int, std::vector<std::tuple<int, double, double>>> divergences;

    for (int r : r_values) {
        double lgamma_r = std::lgamma(r);
        for (double p : p_values) {
            for (int k = 0; k <= max_k; ++k) {
                double base_value = nb_base(k, r, p);
                // double optimized_value = negative_binomial_pmf_optimized(k,
                // r, p, lgamma_r);
                double optimized_value =
                    negative_binomial_pmf_stirling(k, r, p);
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
        std::cout << std::setw(10) << "k" << std::setw(10) << "r"
                  << std::setw(12) << "p" << std::setw(15) << "Difference"
                  << "\n";

        for (const auto &[k, entries] : divergences) {
            for (const auto &[r, p, diff] : entries) {
                std::cout << std::setw(10) << k << std::setw(10) << r
                          << std::setw(12) << p << std::setw(15) << diff
                          << "\n";
            }
        }

        std::cout << std::endl;

        std::cout << "Top 10 divergences:\n";

        std::cout << std::setw(10) << "k" << std::setw(10) << "r"
                  << std::setw(12) << "p" << std::setw(15) << "Difference"
                  << "\n";

        // Temporary vector to hold all divergences (k, r, p, diff)
        std::vector<std::tuple<int, int, double, double>> flat_divergences;

        // Flatten the map into a single vector
        for (const auto &[k, entries] : divergences) {
            for (const auto &[r, p, diff] : entries) {
                flat_divergences.emplace_back(k, r, p, diff);
            }
        }

        // Sort the flattened vector by the difference in descending order
        std::sort(flat_divergences.begin(), flat_divergences.end(),
                  [](const auto &a, const auto &b) {
                      return std::get<3>(a) > std::get<3>(b);
                  });

        size_t limit = 10;

        // Print the top 10 divergences
        for (size_t i = 0; i < std::min(limit, flat_divergences.size()); ++i) {
            const auto &[k, r, p, diff] = flat_divergences[i];
            std::cout << std::setw(10) << k << std::setw(10) << r
                      << std::setw(12) << p << std::setw(15) << diff << "\n";
        }
    }
}
