#pragma once

#include <cmath>

#include "base_nb.hpp"
#include "eigen_nb.hpp"
#include "eigen_nb_cdf.hpp"
#include "eigen_nb_jac.hpp"
#include "utils.hpp"
#include "vector_nb.hpp"

double compute_log_comb_sterling(int k, int r);

double negative_binomial_pmf_expansion(int k, int r, double p);

double negative_binomial_pmf_stirling(int k, int r, double p);
