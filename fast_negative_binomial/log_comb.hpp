#pragma once

#include <cmath>

template <typename T>
double compute_log_comb(int k, T r) {
    if (k < 0 || r <= 0) {
        return -std::numeric_limits<double>::infinity();
    }

    return std::lgamma(k + r) - std::lgamma(r) - std::lgamma(k + 1);
}

double compute_log_comb(int k, int r, double lgamma_r) {
    double log_comb = 0.0;

    if (k < 0 || r <= 0) {
        return -std::numeric_limits<double>::infinity();
    }

    return std::lgamma(k + r) - lgamma_r - std::lgamma(k + 1);

    return log_comb;
}
