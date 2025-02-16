#pragma once

#include <cmath>
#include <iostream>

class LgammaCache {
   public:
    LgammaCache() : current_k_(0), current_lgamma_(0.0), relative_cost_(10) {}

    // Compute lgamma(x) with caching for sorted x
    // double lgamma(double x) {
    //     if (x < 0) {
    //         std::cerr << "Error: x must be positive integer." << std::endl;
    //         return std::numeric_limits<double>::quiet_NaN();
    //     }

    //     if (x == current_k_) {
    //         return current_lgamma_;
    //     } else if (x > current_k_ + relative_cost_) {
    //         current_lgamma_ = std::lgamma(static_cast<double>(x));
    //         current_k_ = x;

    //         return current_lgamma_;

    //     } else {
    //         // Compute iteratively from current_k_ to x
    //         while (current_k_ < x) {
    //             if (current_k_ == 0) {
    //                 // Initialize lgamma(1) = 0
    //                 current_k_ = 1;
    //                 current_lgamma_ = 0.0;
    //             } else {
    //                 // Use the identity: lgamma(x + 1) = log(x) + lgamma(x)
    //                 current_lgamma_ +=
    //                     std::log(static_cast<double>(current_k_));
    //                 current_k_++;
    //             }
    //         }

    //         return current_lgamma_;
    //     }
    // }

    double lgamma(double x) { return std::lgamma(x); }

   private:
    double current_k_;       // The current largest k computed
    double current_lgamma_;  // The current lgamma(k) value
    int relative_cost_;
};
