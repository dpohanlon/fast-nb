#include <Eigen/Dense>
#include <boost/math/distributions/negative_binomial.hpp>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#ifdef ENABLE_BENCHMARK
#include <benchmark/benchmark.h>
#endif

#include "fast_nb.cpp"
#include "test.cpp"
#include "utils.hpp"

std::vector<int> get_poisson(int n, double lambda = 100) {
    std::vector<int> k_vals;
    k_vals.reserve(n);

    static std::mt19937 gen(42);
    std::poisson_distribution<int> poisson_dist(lambda);

    // Generate Poisson-distributed k_vals
    for (int i = 0; i < n; ++i) {
        k_vals.emplace_back(poisson_dist(gen));
    }

    return k_vals;
}

#ifdef ENABLE_BENCHMARK

// --benchmark_time_unit=us

// static void BM_NegativeBinomialPMF(benchmark::State& state) {
//     int r = 50;
//     float p = 0.7;

//     int num_samples = state.range(0);

//     std::vector<int> k_vals = get_poisson(num_samples);

//     // double lgamma_r = std::lgamma(r);

//     for (auto _ : state) {

//         // std::vector<double> results = nb_base_vec(k_vals, r, p);
//         std::vector<double> results = nb_boost_vec(k_vals, r, p);

//         benchmark::DoNotOptimize(results);
//     }
//     state.SetComplexityN(state.range(0));
// }

static void BM_NegativeBinomialPMF(benchmark::State &state) {
    int r = 50;
    float p = 0.7;
    // std::vector<int> k_vals(state.range(0));
    // std::iota(k_vals.begin(), k_vals.end(), 0);

    std::vector<int> k_vals = get_poisson(state.range(0));

    double lgamma_r = std::lgamma(r);

    auto k_vals_eigen = stdVectorToEigenCopy(k_vals);

    // Eigen::VectorXi k = Eigen::VectorXi::LinSpaced(state.range(0), 0,
    // state.range(0));

    for (auto _ : state) {
        // std::vector<float> results(k_vals.size());

        // for (size_t i = 0; i < k_vals.size(); ++i) {
        //     results[i] = boost::math::pdf(boost::math::negative_binomial(r,
        //     p), k_vals[i]);
        //     // results[i] = nb_base(k_vals[i], r, p);
        //     // results[i] = negative_binomial_pmf_lut(k_vals[i], r, p);
        //     // results[i] = negative_binomial_pmf_optimized(k_vals[i], r, p,
        //     lgamma_r);
        // }

        Eigen::VectorXd results = nb2_base_vec_eigen_blocks_no_copy(k_vals_eigen, r, p);

        benchmark::DoNotOptimize(results);
    }
    state.SetComplexityN(state.range(0));
}

// Blocks do help a bit
// static void BM_NegativeBinomialPMF(benchmark::State& state) {
//     int r = 100;
//     float p = 0.7;

//     Eigen::VectorXi k = Eigen::VectorXi::LinSpaced(state.range(0), 0,
//     state.range(0));

//     for (auto _ : state) {

//         Eigen::VectorXd results = nb_base_vec_eigen(k, r, p);

//         benchmark::DoNotOptimize(results);
//     }
//     state.SetComplexityN(state.range(0));
// }

BENCHMARK(BM_NegativeBinomialPMF)->Range(16, 1 << 22)->Complexity();

BENCHMARK_MAIN();

#else

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

int main() {
    int r = 10;
    float p = 0.7;

    std::vector<int> k_vals = get_poisson(10);

    double lgamma_r = std::lgamma(r);

    auto k_vals_eigen = stdVectorToEigenCopy(k_vals);

    for (auto k : k_vals) {
        std::cout << k << std::endl;
    }
    std::cout << std::endl;

    nb_base_vec_eigen_sorted(k_vals_eigen, r, p);

    return 0;
}

#endif
