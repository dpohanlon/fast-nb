import matplotlib.pyplot as plt

import numpy as np
import timeit

# import torch
# import pyro.distributions as dist_pyro
# https://github.com/pytorch/pytorch/issues/121101

from scipy.stats import nbinom
import numpyro.distributions as dist_numpyro

import os

import jax
jax.config.update("jax_enable_x64", True)

from fast_negative_binomial import  negative_binomial2, negative_binomial_boost_vec

os.environ["OMP_NUM_THREADS"] = "4"         # Specify the number of threads
os.environ["OMP_DYNAMIC"] = "FALSE"        # Disable dynamic adjustment of threads
os.environ["OMP_PROC_BIND"] = "TRUE"        # Enable thread affinity
os.environ["OMP_PLACES"] = "cores"          # Define where threads are placed

print("OMP_NUM_THREADS:", os.getenv("OMP_NUM_THREADS"))
print("OMP_DYNAMIC:", os.getenv("OMP_DYNAMIC"))
print("OMP_PROC_BIND:", os.getenv("OMP_PROC_BIND"))
print("OMP_PLACES:", os.getenv("OMP_PLACES"))

r = 1.
m = 50.
repetitions = 100

ks = np.ascontiguousarray(nbinom.rvs(r, r / (m + r), size=100000).astype(np.int32))

plt.hist(ks, bins = 100)
plt.savefig('hist.png')
plt.clf()

j_func = jax.jit(jax.scipy.stats.nbinom.pmf)

def bench_jax_scipy():
    p = r / (m + r)
    j_func(ks, r, p).block_until_ready()

def bench_fast_nb():
    negative_binomial2(ks, m, r)
    # negative_binomial2_vec(ks, m, r)

def bench_boost_nb():
    p = r / (m + r)
    negative_binomial_boost_vec(ks, r, p)

def bench_scipy():
    p = r / (m + r)
    nbinom.pmf(ks, r, p)

# ks_t = torch.tensor(ks, dtype=torch.float64)
# def bench_pyro():
#     p = r / (m + r)
#     dist_pyro.NegativeBinomial(r, p).log_prob(ks_t)

def bench_numpyro():
    p = r / (m + r)
    dist_numpyro.NegativeBinomialProbs(r, p).log_prob(ks)

for f in [bench_jax_scipy, bench_fast_nb, bench_scipy, bench_numpyro, bench_boost_nb]:

    ks = np.ascontiguousarray(nbinom.rvs(r, r / (m + r), size=100000).astype(np.int32))

    elapsed_time = timeit.timeit(f, number=repetitions)

    average_time = elapsed_time / repetitions
    average_time *= 1e3  # Convert to milliseconds

    print(f"Average {f.__name__} computation time over {repetitions} runs: {average_time:.6f} ms")

scipy_nb_res = nbinom.pmf(ks, r, r / (m + r))
fast_nb_res = negative_binomial2(ks, m, r)

plt.plot(ks, fast_nb_res - scipy_nb_res, '.')
plt.savefig('difference.png')
plt.clf()
