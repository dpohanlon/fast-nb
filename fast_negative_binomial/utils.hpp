#pragma once

// Precompute log combinatorial coefficients for small values of k, r.
// This is only constexpr in GCC, but will be for both in C++26
#if defined(__GNUC__) && !defined(__clang__)
    #include "log_comb_gcc.hpp"
#else
    #include "log_comb.hpp"
#endif

// Threshold for switching to Stirling's approximation
const double STIRLING_THRESHOLD = 10.0;

double prob(double mean, double concentration)
{
    double km = mean + concentration;

    if (km <= 0) {
        return 0.;
    }

    return concentration / km;
}

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
