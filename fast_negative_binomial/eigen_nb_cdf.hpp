#pragma once

#include <omp.h>

#include <Eigen/Dense>
#include <boost/math/special_functions/beta.hpp>
#include <boost/sort/sort.hpp>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <cmath>
#include <numeric>

#include "utils.hpp"

inline double nb_cdf_single(int k, double r, double p) {
    return boost::math::ibeta(r, static_cast<double>(k) + 1.0, p);
}

inline double nb2_cdf_single(int k, double mean, double concentration) {
    double p = concentration / (mean + concentration);
    return nb_cdf_single(k, concentration, p);
}

inline double zinb2_cdf_single(int k, double mean, double concentration, double alpha) {
    double p = concentration / (mean + concentration);
    double nb_cdf = boost::math::ibeta(concentration, static_cast<double>(k) + 1.0, p);
    return alpha + (1.0 - alpha) * nb_cdf;
}

inline void compute_cdf_block(const FixedVectorXi &k_block,
                              FixedVectorXd &cdf_block, double r, double p) {
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        if (i > 0 && k_block[i] == k_block[i - 1]) {
            cdf_block[i] = cdf_block[i - 1];
        } else {
            // CDF(r, k, p) = ibeta(r, k+1, p)
            double rr = static_cast<double>(r);
            double kk = static_cast<double>(k_block[i]) + 1.0;
            cdf_block[i] = boost::math::ibeta(rr, kk, p);
        }
    }
}

void process_cdf_blocks(const Eigen::VectorXi &k, Eigen::VectorXd &results,
                        double r, double p, int num_blocks) {
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int block = 0; block < num_blocks; ++block) {
            int start = block * BLOCK_SIZE;
            FixedVectorXi k_block =
                Eigen::Map<const FixedVectorXi>(k.data() + start);

            FixedVectorXd cdf_block;
            compute_cdf_block(k_block, cdf_block, r, p);

            Eigen::Map<FixedVectorXd>(results.data() + start) = cdf_block;
        }
    }
}

Eigen::VectorXd process_cdf_remaining(const Eigen::VectorXi &k, int start,
                                      int remaining, double r, double p) {
    Eigen::VectorXd out(remaining);
    for (int i = 0; i < remaining; ++i) {
        if (i > 0 && k[start + i] == k[start + i - 1]) {
            out[i] = out[i - 1];
        } else {
            double rr = static_cast<double>(r);
            double kk = static_cast<double>(k[start + i]) + 1.0;
            out[i] = boost::math::ibeta(rr, kk, p);
        }
    }
    return out;
}

template <typename T>
Eigen::VectorXd nb_cdf_vec_eigen_blocks_no_copy(Eigen::Ref<Eigen::VectorXi> k,
                                                T r, double p) {
    // Sort k in-place
    boost::sort::parallel_stable_sort(k.data(), k.data() + k.size());

    Eigen::VectorXd results(k.size());
    int num_blocks = k.size() / BLOCK_SIZE;
    int remaining = k.size() % BLOCK_SIZE;

    // Process complete blocks
    if (num_blocks > 0) {
        process_cdf_blocks(k, results, r, p, num_blocks);
    }

    // Process any remainder
    if (remaining > 0) {
        int start = num_blocks * BLOCK_SIZE;
        Eigen::VectorXd tail = process_cdf_remaining(k, start, remaining, r, p);
        results.segment(start, remaining) = tail;
    }

    return results;
}

template <typename T>
Eigen::VectorXd nb_cdf_vec_eigen_blocks(const Eigen::VectorXi &k_in, T r,
                                        double p) {
    Eigen::VectorXi k_copy = k_in;
    return nb_cdf_vec_eigen_blocks_no_copy(k_copy, r, p);
}

Eigen::VectorXd nb2_cdf_vec_eigen_blocks(const Eigen::VectorXi &k, double m,
                                         double r) {
    double p = prob(m, r);
    return nb_cdf_vec_eigen_blocks(k, r, p);
}

Eigen::VectorXd nb2_cdf_vec_eigen_blocks_no_copy(Eigen::Ref<Eigen::VectorXi> k,
                                                 double m, double r) {
    double p = prob(m, r);
    return nb_cdf_vec_eigen_blocks_no_copy(k, r, p);
}

Eigen::VectorXd zinb2_cdf_vec_eigen_blocks(const Eigen::VectorXi &k, double m,
                                           double r, double alpha) {
    double p = prob(m, r);
    Eigen::VectorXi k_copy = k;

    Eigen::VectorXd cdf = nb_cdf_vec_eigen_blocks_no_copy(k_copy, r, p);

    Eigen::VectorXd zinb_cdf =
        Eigen::VectorXd::Constant(cdf.size(), alpha) + (1.0 - alpha) * cdf;

    return zinb_cdf;
}

inline double nb2_cdf_single_exposure(int k, double mu0, double r, double exposure) {
    const double m = mu0 * exposure;
    const double p = prob(m, r);
    return nb_cdf_single(k, r, p);  // CDF = I_p(r, k+1)
}

inline double zinb2_cdf_single_exposure(int k, double mu0, double r, double alpha, double exposure) {
    const double m = mu0 * exposure;
    const double p = prob(m, r);
    const double nb_cdf = nb_cdf_single(k, r, p);
    return alpha + (1.0 - alpha) * nb_cdf;
}

inline void compute_cdf_block_exposure(const FixedVectorXi &k_block,
                                       const FixedVectorXd &exposure_block,
                                       FixedVectorXd &cdf_block,
                                       double r, double mu0) {
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        const double m = mu0 * exposure_block[i];
        const double p = prob(m, r);
        // CDF(r, k, p) = ibeta(r, k+1, p)
        const double rr = static_cast<double>(r);
        const double kk = static_cast<double>(k_block[i]) + 1.0;
        cdf_block[i] = boost::math::ibeta(rr, kk, p);
    }
}

inline void compute_zinb_cdf_block_exposure(const FixedVectorXi &k_block,
                                            const FixedVectorXd &exposure_block,
                                            FixedVectorXd &cdf_block,
                                            double r, double mu0, double alpha) {
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        const double m = mu0 * exposure_block[i];
        const double p = prob(m, r);
        const double rr = static_cast<double>(r);
        const double kk = static_cast<double>(k_block[i]) + 1.0;
        const double nb_cdf = boost::math::ibeta(rr, kk, p);
        cdf_block[i] = alpha + (1.0 - alpha) * nb_cdf;
    }
}

inline Eigen::VectorXd process_cdf_remaining_exposure(const Eigen::VectorXi &k,
                                                      const Eigen::VectorXd &exposure,
                                                      int start, int remaining,
                                                      double r, double mu0) {
    Eigen::VectorXd out(remaining);
    for (int i = 0; i < remaining; ++i) {
        const double m = mu0 * exposure[start + i];
        const double p = prob(m, r);
        const double rr = static_cast<double>(r);
        const double kk = static_cast<double>(k[start + i]) + 1.0;
        out[i] = boost::math::ibeta(rr, kk, p);
    }
    return out;
}

inline Eigen::VectorXd process_zinb_cdf_remaining_exposure(const Eigen::VectorXi &k,
                                                           const Eigen::VectorXd &exposure,
                                                           int start, int remaining,
                                                           double r, double mu0, double alpha) {
    Eigen::VectorXd out(remaining);
    for (int i = 0; i < remaining; ++i) {
        const double m = mu0 * exposure[start + i];
        const double p = prob(m, r);
        const double rr = static_cast<double>(r);
        const double kk = static_cast<double>(k[start + i]) + 1.0;
        const double nb_cdf = boost::math::ibeta(rr, kk, p);
        out[i] = alpha + (1.0 - alpha) * nb_cdf;
    }
    return out;
}

inline Eigen::VectorXd nb2_cdf_vec_eigen_exposure(const Eigen::VectorXi &k,
                                                  double mu0, double r,
                                                  const Eigen::VectorXd &exposure) {
    const int n = static_cast<int>(k.size());
    Eigen::VectorXd results(n);

    // Process complete blocks
    const int num_blocks = n / BLOCK_SIZE;
    if (num_blocks > 0) {
        #pragma omp parallel for schedule(static)
        for (int block = 0; block < num_blocks; ++block) {
            const int start = block * BLOCK_SIZE;

            FixedVectorXi k_block = Eigen::Map<const FixedVectorXi>(k.data() + start);
            FixedVectorXd e_block = Eigen::Map<const FixedVectorXd>(exposure.data() + start);

            FixedVectorXd cdf_block;
            compute_cdf_block_exposure(k_block, e_block, cdf_block, r, mu0);

            Eigen::Map<FixedVectorXd>(results.data() + start) = cdf_block;
        }
    }

    // Process remainder
    const int remaining = n % BLOCK_SIZE;
    if (remaining > 0) {
        const int start = num_blocks * BLOCK_SIZE;
        Eigen::VectorXd tail = process_cdf_remaining_exposure(k, exposure, start, remaining, r, mu0);
        results.segment(start, remaining) = tail;
    }

    return results;
}

inline Eigen::VectorXd zinb2_cdf_vec_eigen_exposure(const Eigen::VectorXi &k,
                                                    double mu0, double r, double alpha,
                                                    const Eigen::VectorXd &exposure) {
    const int n = static_cast<int>(k.size());
    Eigen::VectorXd results(n);

    const int num_blocks = n / BLOCK_SIZE;
    if (num_blocks > 0) {
        #pragma omp parallel for schedule(static)
        for (int block = 0; block < num_blocks; ++block) {
            const int start = block * BLOCK_SIZE;

            FixedVectorXi k_block = Eigen::Map<const FixedVectorXi>(k.data() + start);
            FixedVectorXd e_block = Eigen::Map<const FixedVectorXd>(exposure.data() + start);

            FixedVectorXd cdf_block;
            compute_zinb_cdf_block_exposure(k_block, e_block, cdf_block, r, mu0, alpha);

            Eigen::Map<FixedVectorXd>(results.data() + start) = cdf_block;
        }
    }

    const int remaining = n % BLOCK_SIZE;
    if (remaining > 0) {
        const int start = num_blocks * BLOCK_SIZE;
        Eigen::VectorXd tail = process_zinb_cdf_remaining_exposure(k, exposure, start, remaining, r, mu0, alpha);
        results.segment(start, remaining) = tail;
    }

    return results;
}
