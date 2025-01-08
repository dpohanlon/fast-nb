#pragma once

#include <cmath>

template <typename T>
double compute_log_comb(int k, T r) {
    if (k < 0 || r <= 0) {
        return -std::numeric_limits<double>::infinity();
    }

    return std::lgamma(k + r) - std::lgamma(r) - std::lgamma(k + 1);
}

// double compute_log_comb(int k, int r, double lgamma_r) {
//     double log_comb = 0.0;

//     // Sum log(k + 1) to log(k + r - 1)
//     for(int i = 1; i < r; ++i){
//         log_comb += std::log(static_cast<double>(k) +
//         static_cast<double>(i));
//     }

//     // Subtract precomputed lgamma(r)
//     log_comb -= lgamma_r;

//     return log_comb;
// }

double compute_log_comb(int k, int r, double lgamma_r) {
    double log_comb = 0.0;

    if (k < 0 || r <= 0) {
        return -std::numeric_limits<double>::infinity();
    }

    return std::lgamma(k + r) - lgamma_r - std::lgamma(k + 1);

    return log_comb;
}
