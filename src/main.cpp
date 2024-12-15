#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

#ifdef ENABLE_BENCHMARK
#include <benchmark/benchmark.h>
#endif

#include "fast_nb.cpp"
#include "test.cpp"

#ifdef ENABLE_BENCHMARK

static void BM_NegativeBinomialPMF(benchmark::State& state) {
    int r = 5;
    float p = 0.7;
    std::vector<int> k_vals(state.range(0));
    std::iota(k_vals.begin(), k_vals.end(), 0);

    // float lgamma_r = precompute_lgamma_r(r);

    for (auto _ : state) {
        std::vector<float> results(k_vals.size());
        for (size_t i = 0; i < k_vals.size(); ++i) {
            results[i] = negative_binomial_pmf_base(k_vals[i], r, p);
            // results[i] = negative_binomial_pmf_lut(k_vals[i], r, p);
            // results[i] = negative_binomial_pmf_optimized(k_vals[i], r, p, lgamma_r);
        }
        benchmark::DoNotOptimize(results);
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_NegativeBinomialPMF)->Range(16, 1 << 20)->Complexity();

BENCHMARK_MAIN();

#else

int main() {
    // Tolerance for comparison
    double tolerance = 1e-6;

    // Maximum value of k (number of failures)
    int max_k = 100;

    // Set of r values (number of successes)
    std::vector<int> r_values = {1, 5, 10, 20, 50, 100, 200, 500, 1000};

    // Set of p values (success probabilities)
    std::vector<double> p_values = {0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99};

    // Run the accuracy test
    test_accuracy(tolerance, max_k, r_values, p_values);

    return 0;
}

#endif
