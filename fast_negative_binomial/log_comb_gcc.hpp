#include <cmath>
#include <array>
#include <iostream>

constexpr int MAX_K = 85;
constexpr int MAX_R = 85;

constexpr double log_comb(int k, int r) {
    if (k < 0 || r <= 0) {
        return 0.0;
    }

    return std::log(std::tgamma(static_cast<double>(k + r))) - std::log(std::tgamma(static_cast<double>(r))) - std::log(std::tgamma(static_cast<double>(k + 1)));
}

constexpr std::array<std::array<double, MAX_R + 1>, MAX_K + 1> precompute_log_comb() {
    std::array<std::array<double, MAX_R + 1>, MAX_K + 1> table = {};

    for (int k = 0; k <= MAX_K; ++k) {
        for (int r = 0; r <= MAX_R; ++r) {
            if (r == 0) {
                table[k][r] = 0.0;
            } else {
                table[k][r] = log_comb(k, r);
            }
        }
    }
    return table;
}

constexpr auto LOG_COMB_TABLE = precompute_log_comb();

constexpr double compute_log_comb(int k, int r) {
    // Ensure that k and r are non-negative and r > 0
    if(k < 0 || r <= 0){
        return -std::numeric_limits<double>::infinity(); // Handle invalid cases appropriately
    }

    if (k <= MAX_K && r <= MAX_R) {
        return LOG_COMB_TABLE[k][r];
    } else {
        return std::lgamma(k + r) - std::lgamma(r) - std::lgamma(k + 1);
    }

}
