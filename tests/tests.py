import os
import numpy as np
from scipy.stats import nbinom

from fast_negative_binomial import  negative_binomial2

os.environ["OMP_NUM_THREADS"] = "8"         # Specify the number of threads
os.environ["OMP_DYNAMIC"] = "FALSE"        # Disable dynamic adjustment of threads
os.environ["OMP_PROC_BIND"] = "TRUE"        # Enable thread affinity
os.environ["OMP_PLACES"] = "cores"          # Define where threads are placed

print("OMP_NUM_THREADS:", os.getenv("OMP_NUM_THREADS"))
print("OMP_DYNAMIC:", os.getenv("OMP_DYNAMIC"))
print("OMP_PROC_BIND:", os.getenv("OMP_PROC_BIND"))
print("OMP_PLACES:", os.getenv("OMP_PLACES"))

def test_nb2():

    n = 100

    ms = np.linspace(1, 100, n)
    rs = np.linspace(1, 100, n)

    for i in range(n):

        m = ms[i]
        r = rs[i]

        ks = nbinom.rvs(r, r / (m + r), size=1000).astype(np.int32)

        res = negative_binomial2(ks, m, r)

        scipy_nb_res = nbinom.pmf(ks, r, r / (m + r))

        assert np.allclose(res, scipy_nb_res), f"Failed for m={m}, r={r}"

if __name__ == "__main__":
    test_nb2()
