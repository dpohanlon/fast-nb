#pragma once

#include <omp.h>

#include <Eigen/Dense>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/sort/sort.hpp>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <cmath>
#include <numeric>

#include "eigen_nb.hpp"
#include "utils.hpp"

// Compute gradients block-wise for d(logNB)/dp and d(logNB)/dr.
//
//   d(logNB)/dp =  r/p  -  k/(1-p)
//   d(logNB)/dr =  log(p)  -  digamma(r)  +  digamma(k + r)
//
// We reuse the previously computed gradient if k_block[i] repeats.
inline void compute_grad_block(const FixedVectorXi &k_block,
                               FixedVectorXd &grad_p_block,
                               FixedVectorXd &grad_r_block, double r, double p,
                               double digamma_r) {
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        if (i > 0 && k_block[i] == k_block[i - 1]) {
            grad_p_block[i] = grad_p_block[i - 1];
            grad_r_block[i] = grad_r_block[i - 1];
        } else {
            double val_kr =
                boost::math::digamma(static_cast<double>(k_block[i]) + r);
            grad_p_block[i] =
                r / p - static_cast<double>(k_block[i]) / (1.0 - p);
            grad_r_block[i] = std::log(p) - digamma_r + val_kr;
        }
    }
}

// Processes all complete blocks in parallel for the NB gradient.
void process_grad_blocks(const Eigen::VectorXi &k, Eigen::VectorXd &grad_p,
                         Eigen::VectorXd &grad_r, double r, double p,
                         int num_blocks) {
    double digamma_r = boost::math::digamma(r);

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (int block = 0; block < num_blocks; ++block) {
            int start = block * BLOCK_SIZE;
            FixedVectorXi k_block =
                Eigen::Map<const FixedVectorXi>(k.data() + start);

            FixedVectorXd grad_p_block, grad_r_block;
            compute_grad_block(k_block, grad_p_block, grad_r_block, r, p,
                               digamma_r);

            Eigen::Map<FixedVectorXd>(grad_p.data() + start) = grad_p_block;
            Eigen::Map<FixedVectorXd>(grad_r.data() + start) = grad_r_block;
        }
    }
}

Eigen::MatrixXd process_grad_remaining(const Eigen::VectorXi &k, int start,
                                       int remaining, double r, double p) {
    Eigen::MatrixXd out(remaining, 2);

    double digamma_r = boost::math::digamma(r);
    for (int i = 0; i < remaining; ++i) {
        if (i > 0 && k[start + i] == k[start + i - 1]) {
            out(i, 0) = out(i - 1, 0);
            out(i, 1) = out(i - 1, 1);
        } else {
            double val_kr =
                boost::math::digamma(static_cast<double>(k[start + i]) + r);
            out(i, 0) = r / p - static_cast<double>(k[start + i]) / (1.0 - p);
            out(i, 1) = std::log(p) - digamma_r + val_kr;
        }
    }

    return out;
}

template <typename T>
Eigen::MatrixXd log_nb_gradient_vec_eigen_blocks_no_copy(
    Eigen::Ref<Eigen::VectorXi> k, T r, double p) {
    // Sort k in-place
    boost::sort::parallel_stable_sort(k.data(), k.data() + k.size());
    // boost::sort::flat_stable_sort(k.data(), k.data() + k.size());

    // Prepare storage
    Eigen::MatrixXd grad(k.size(), 2);
    Eigen::VectorXd grad_p(k.size()), grad_r(k.size());

    const int num_blocks = k.size() / BLOCK_SIZE;
    const int remaining = k.size() % BLOCK_SIZE;

    // Process complete blocks in parallel
    if (num_blocks > 0) {
        process_grad_blocks(k, grad_p, grad_r, r, p, num_blocks);
    }

    // Process the remainder
    if (remaining > 0) {
        int start = num_blocks * BLOCK_SIZE;
        Eigen::MatrixXd tail =
            process_grad_remaining(k, start, remaining, r, p);
        grad_p.segment(start, remaining) = tail.col(0);
        grad_r.segment(start, remaining) = tail.col(1);
    }

    // Combine into a single MatrixXd
    grad.col(0) = grad_p;
    grad.col(1) = grad_r;

    return grad;
}

template <typename T>
Eigen::MatrixXd log_nb_gradient_vec_eigen_blocks(const Eigen::VectorXi &k_in,
                                                 T r, double p) {
    Eigen::VectorXi k_copy = k_in;
    return log_nb_gradient_vec_eigen_blocks_no_copy(k_copy, r, p);
}

Eigen::MatrixXd log_nb2_gradient_vec_eigen_blocks(const Eigen::VectorXi &k,
                                                  double m, double r) {
    // m : mean
    // r : concentration

    double p = prob(m, r);

    Eigen::MatrixXd jac = log_nb_gradient_vec_eigen_blocks(k, r, p);

    // Chain rule to go from d(logNB)/dp to d(logNB2)/dm

    double dp_dm = -r / ((m + r) * (m + r));

    jac.col(0) = jac.col(0).array() * dp_dm;

    return jac;
}

Eigen::MatrixXd log_nb2_gradient_vec_eigen_blocks_no_copy(
    Eigen::Ref<Eigen::VectorXi> k, double m, double r) {
    // m : mean
    // r : concentration

    double p = prob(m, r);

    Eigen::MatrixXd jac = log_nb_gradient_vec_eigen_blocks_no_copy(k, r, p);

    // Chain rule to go from d(logNB)/dp to d(logNB2)/dm

    double dp_dm = -r / ((m + r) * (m + r));

    jac.col(0) = jac.col(0).array() * dp_dm;

    return jac;
}

Eigen::MatrixXd log_zinb_gradient_vec_eigen_blocks_post_process_select(
    const Eigen::VectorXi &k_in, double m, double r, double alpha) {

    const double eps_prob = 1e-300;
    const double eps_one  = 1e-12;
    const double eps_a    = 1e-8;

    double a = std::clamp(alpha, eps_a, 1.0 - eps_a);

    double p = prob(m, r);

    Eigen::MatrixXd nb_grad = log_nb_gradient_vec_eigen_blocks(k_in, r, p);

    // Chain rule to go from d(logNB)/dp to d(logNB2)/dm

    double dp_dm = -r / ((m + r) * (m + r));
    nb_grad.col(0) = nb_grad.col(0).array() * dp_dm;

    Eigen::VectorXd nb_probs = nb_base_vec_eigen_blocks(k_in, r, p);

    nb_probs = nb_probs.array().max(eps_prob).min(1.0 - eps_one);

    Eigen::MatrixXd zinb_grad(k_in.size(), 3);
    zinb_grad.leftCols(2) = nb_grad;

    Eigen::ArrayXd denom_zero = a + (1.0 - a) * nb_probs.array();
    denom_zero = denom_zero.max(eps_one);

    Eigen::VectorXd alpha_grad_if_zero =
        (1.0 - nb_probs.array()) / denom_zero;

    double denom_nz = (1.0 - a);
    if (denom_nz < eps_one) denom_nz = eps_one;

    Eigen::VectorXd alpha_grad_if_nonzero =
        Eigen::VectorXd::Constant(k_in.size(), -1.0 / denom_nz);

    zinb_grad.col(2) =
        (k_in.array() == 0).select(alpha_grad_if_zero, alpha_grad_if_nonzero);

    return zinb_grad;
}

inline void process_grad_remaining_sumcheck(
    const Eigen::VectorXi &k,
    int start,
    int remaining,
    double r,
    double p,
    double &sum_p,
    double &sum_r,
    int &any_nonfinite
) {
    const double digamma_r = boost::math::digamma(r);

    double sp = 0.0;
    double sr = 0.0;
    int bad = 0;

    double last_gp = 0.0;
    double last_gr = 0.0;
    bool have_last = false;

    for (int i = 0; i < remaining; ++i) {
        const int idx = start + i;

        double gp, gr;
        if (have_last && i > 0 && k[idx] == k[idx - 1]) {
            gp = last_gp;
            gr = last_gr;
        } else {
            const double val_kr =
                boost::math::digamma(static_cast<double>(k[idx]) + r);
            gp = r / p - static_cast<double>(k[idx]) / (1.0 - p);
            gr = std::log(p) - digamma_r + val_kr;
            last_gp = gp;
            last_gr = gr;
            have_last = true;
        }

        sp += gp;
        sr += gr;
        bad |= (!std::isfinite(gp) || !std::isfinite(gr));
    }

    sum_p = sp;
    sum_r = sr;
    any_nonfinite = bad;
}

inline void compute_grad_block_sumcheck(
    const FixedVectorXi &k_block,
    double r,
    double p,
    double digamma_r,
    double &sum_p,
    double &sum_r,
    int &any_nonfinite
) {
    double sp = 0.0;
    double sr = 0.0;
    int bad = 0;

    double last_gp = 0.0;
    double last_gr = 0.0;
    bool have_last = false;

    for (int i = 0; i < BLOCK_SIZE; ++i) {
        double gp, gr;

        if (have_last && i > 0 && k_block[i] == k_block[i - 1]) {
            gp = last_gp;
            gr = last_gr;
        } else {
            const double val_kr =
                boost::math::digamma(static_cast<double>(k_block[i]) + r);
            gp = r / p - static_cast<double>(k_block[i]) / (1.0 - p);
            gr = std::log(p) - digamma_r + val_kr;
            last_gp = gp;
            last_gr = gr;
            have_last = true;
        }

        sp += gp;
        sr += gr;
        bad |= (!std::isfinite(gp) || !std::isfinite(gr));
    }

    sum_p = sp;
    sum_r = sr;
    any_nonfinite = bad;
}


inline void process_grad_blocks_sumcheck(
    const Eigen::VectorXi &k,
    double r,
    double p,
    int num_blocks,
    double &sum_grad_p,
    double &sum_grad_r,
    int &any_nonfinite
) {
    const double digamma_r = boost::math::digamma(r);

    double sp = 0.0;
    double sr = 0.0;
    int bad = 0;

    #pragma omp parallel for schedule(static) reduction(+:sp,sr) reduction(|:bad)
    for (int block = 0; block < num_blocks; ++block) {
        const int start = block * BLOCK_SIZE;
        const Eigen::Map<const FixedVectorXi> k_block(k.data() + start);

        double local_sp = 0.0, local_sr = 0.0;
        int local_bad = 0;

        compute_grad_block_sumcheck(k_block, r, p, digamma_r, local_sp, local_sr, local_bad);

        sp += local_sp;
        sr += local_sr;
        bad |= local_bad;
    }

    sum_grad_p = sp;
    sum_grad_r = sr;
    any_nonfinite = bad;
}

template <typename T>
inline std::pair<Eigen::Vector2d, bool> log_nb_gradient_means_eigen_blocks_no_copy(
    Eigen::Ref<Eigen::VectorXi> k,
    T r,
    double p
) {
    const int n = static_cast<int>(k.size());
    const int num_blocks = n / BLOCK_SIZE;
    const int remaining  = n % BLOCK_SIZE;

    double sum_p = 0.0, sum_r = 0.0;
    int bad = 0;

    if (num_blocks > 0) {
        double bp = 0.0, br = 0.0;
        int bbad = 0;
        process_grad_blocks_sumcheck(k, r, p, num_blocks, bp, br, bbad);
        sum_p += bp;
        sum_r += br;
        bad |= bbad;
    }

    if (remaining > 0) {
        const int start = num_blocks * BLOCK_SIZE;
        double tp = 0.0, tr = 0.0;
        int tbad = 0;
        process_grad_remaining_sumcheck(k, start, remaining, r, p, tp, tr, tbad);
        sum_p += tp;
        sum_r += tr;
        bad |= tbad;
    }

    Eigen::Vector2d means;
    const double inv_n = 1.0 / static_cast<double>(n);
    means[0] = sum_p * inv_n;
    means[1] = sum_r * inv_n;
    return {means, bad != 0};
}

template <typename T>
inline std::pair<Eigen::Vector2d, bool> log_nb_gradient_means_eigen_blocks(
    const Eigen::VectorXi &k_in,
    T r,
    double p
) {
    Eigen::VectorXi k_copy = k_in;
    return log_nb_gradient_means_eigen_blocks_no_copy(k_copy, r, p);
}

inline std::pair<Eigen::Vector3d, bool> log_zinb_gradient_means_eigen_blocks_post_process_select(
    const Eigen::VectorXi &k_in,
    double m,
    double r,
    double alpha
) {
    const double eps_prob = 1e-300;
    const double eps_one  = 1e-12;
    const double eps_a    = 1e-8;

    const double a = std::clamp(alpha, eps_a, 1.0 - eps_a);
    const double p = prob(m, r);
    const int n = static_cast<int>(k_in.size());

    // Means for NB part: mean d/dp, mean d/dr (and bad flag)
    auto nb_means = log_nb_gradient_means_eigen_blocks(k_in, r, p);

    // Chain rule to mean d/dm
    const double dp_dm = -r / ((m + r) * (m + r));
    const double mean_dm = nb_means.first[0] * dp_dm;
    const double mean_dr = nb_means.first[1];

    bool bad = nb_means.second || !std::isfinite(dp_dm);

    // Alpha gradient mean:
    // grad_alpha = (k==0) ? (1 - nb_prob)/(a + (1-a)*nb_prob) : -1/(1-a)
    // We can compute nb_prob on the fly and just accumulate the mean.

    const double denom_nz = std::max(1.0 - a, eps_one);
    const double alpha_grad_nz = -1.0 / denom_nz;

    double sum_alpha = 0.0;

    // This assumes nb_base_vec_eigen_blocks(k_in, r, p) gives per-element NB probabilities.
    // If it internally allocates a vector, this still avoids the big 3-col grad matrix,
    // but for maximum memory savings you'd implement a nb_prob means streaming version too.
    // This is slow
    Eigen::VectorXd nb_probs = nb_base_vec_eigen_blocks(k_in, r, p);
    nb_probs = nb_probs.array().max(eps_prob).min(1.0 - eps_one);

    for (int i = 0; i < n; ++i) {
        const double q = nb_probs[i];
        double ga;
        if (k_in[i] == 0) {
            double denom0 = a + (1.0 - a) * q;
            if (denom0 < eps_one) denom0 = eps_one;
            ga = (1.0 - q) / denom0;
        } else {
            ga = alpha_grad_nz;
        }
        sum_alpha += ga;
        bad = bad || !std::isfinite(ga);
    }

    Eigen::Vector3d means;
    const double inv_n = 1.0 / static_cast<double>(n);
    means[0] = mean_dm;
    means[1] = mean_dr;
    means[2] = sum_alpha * inv_n;

    return {means, bad};
}

inline void compute_grad_block_exposure(
    const FixedVectorXi &k_block,
    const FixedVectorXd &e_block,
    FixedVectorXd &d_mu0_block,
    FixedVectorXd &d_r_block,
    double mu0,
    double r,
    double digamma_r
) {
    // For full blocks only; the remainder is handled separately.
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        const int    ki = k_block[i];
        const double ej = e_block[i];
        const double mj = mu0 * ej;                       // per-sample mean
        const double pj = prob(mj, r);                    // r/(r+mj)
        const double logp = std::log(pj);
        const double digamma_kr = boost::math::digamma(static_cast<double>(ki) + r);

        // d(log NB)/dp and chain to m_j then to mu0 via m_j = mu0 * e_j
        const double dlog_dp = r / pj - static_cast<double>(ki) / (1.0 - pj);
        const double dp_dm   = -r / ((mj + r) * (mj + r));
        d_mu0_block[i] = dlog_dp * dp_dm * ej;

        // d(log NB)/dr at (r, p_j) with p_j a function of m_j (but derivative here w.r.t r only)
        d_r_block[i] = logp - digamma_r + digamma_kr;
    }
}

inline void process_grad_blocks_exposure(
    const Eigen::VectorXi &k,
    const Eigen::VectorXd &exposure,
    Eigen::VectorXd &d_mu0,
    Eigen::VectorXd &d_r,
    double mu0,
    double r,
    int num_blocks
) {
    const double digamma_r = boost::math::digamma(r);

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int block = 0; block < num_blocks; ++block) {
            const int start = block * BLOCK_SIZE;

            const Eigen::Map<const FixedVectorXi> k_block(k.data() + start);
            const Eigen::Map<const FixedVectorXd> e_block(exposure.data() + start);

            FixedVectorXd d_mu0_block, d_r_block;
            compute_grad_block_exposure(k_block, e_block, d_mu0_block, d_r_block, mu0, r, digamma_r);

            Eigen::Map<FixedVectorXd>(d_mu0.data() + start) = d_mu0_block;
            Eigen::Map<FixedVectorXd>(d_r.data()   + start) = d_r_block;
        }
    }
}

inline Eigen::MatrixXd process_grad_remaining_exposure(
    const Eigen::VectorXi &k,
    const Eigen::VectorXd &exposure,
    int start,
    int remaining,
    double mu0,
    double r
) {
    Eigen::MatrixXd out(remaining, 2);
    const double digamma_r = boost::math::digamma(r);

    for (int i = 0; i < remaining; ++i) {
        const int    idx = start + i;
        const int    ki  = k[idx];
        const double ej  = exposure[idx];
        const double mj  = mu0 * ej;
        const double pj  = prob(mj, r);
        const double logp = std::log(pj);
        const double digamma_kr = boost::math::digamma(static_cast<double>(ki) + r);

        const double dlog_dp = r / pj - static_cast<double>(ki) / (1.0 - pj);
        const double dp_dm   = -r / ((mj + r) * (mj + r));

        out(i, 0) = dlog_dp * dp_dm * ej;                  // d/d mu0
        out(i, 1) = logp - digamma_r + digamma_kr;         // d/d r
    }
    return out;
}

inline Eigen::MatrixXd log_nb2_gradient_vec_eigen_exposure(
    const Eigen::VectorXi &k,
    double mu0,
    double r,
    const Eigen::VectorXd &exposure
) {
    // No sorting here — p varies per observation via exposure.
    const int n = static_cast<int>(k.size());

    Eigen::MatrixXd grad(n, 2);
    Eigen::VectorXd d_mu0(n), d_r(n);

    const int num_blocks = n / BLOCK_SIZE;
    const int remaining  = n % BLOCK_SIZE;

    if (num_blocks > 0) {
        process_grad_blocks_exposure(k, exposure, d_mu0, d_r, mu0, r, num_blocks);
    }

    if (remaining > 0) {
        const int start = num_blocks * BLOCK_SIZE;
        Eigen::MatrixXd tail = process_grad_remaining_exposure(k, exposure, start, remaining, mu0, r);
        d_mu0.segment(start, remaining) = tail.col(0);
        d_r.segment(start, remaining)   = tail.col(1);
    }

    grad.col(0) = d_mu0;
    grad.col(1) = d_r;
    return grad;
}
