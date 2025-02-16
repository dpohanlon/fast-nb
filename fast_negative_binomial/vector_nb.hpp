#pragma once

#include <boost/math/distributions/negative_binomial.hpp>
#include <cmath>
#include <vector>

#include "base_nb.hpp"
#include "utils.hpp"

// I can probably template specialise these for vector or scalar, but not sure
// if it's worth it

template <typename T>
std::vector<double> nb_base_vec(std::vector<int> k, T r, double p) {
    double lgamma_r = std::lgamma(static_cast<double>(r));

    std::vector<double> results(k.size());

    for (int i = 0; i < k.size(); ++i) {
        results[i] = nb_base_fixed_r(k[i], r, p, lgamma_r);
    }

    return results;
}

std::vector<double> nb_boost_vec(std::vector<int> k, double r, double p) {
    std::vector<double> results(k.size());

    boost::math::negative_binomial_distribution<> nb(r, p);

    for (int i = 0; i < k.size(); ++i) {
        results[i] = boost::math::pdf(nb, k[i]);
    }

    return results;
}

std::vector<double> nb2_base_vec(std::vector<int> k, double m, double r) {
    // m : mean
    // r : concentration

    double p = prob(m, r);

    return nb_base_vec(k, r, p);
}
