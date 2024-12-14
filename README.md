# fast_nb_test

There are a few general optimisations:

* A lookup table for log-gamma for small values (<= 85) of k and n (only on GCC with `constexpr` cmath)
* Stirling's approximation for large (> 10) values of k and r
* An optimisation when r is fixed that pre-computes these values and re-uses them every invocation of log-gamma

These result in divergences with the high-precision version of at most ~1E-5. There are some more extreme approximations that can be made:

* Expansion form of `std::exp`
* Expansion form of `std::log`

This is mostly envisioned for calculations where r and p are scalars and k is a vector of reasonable size (~1000). As such the vectorisation is performed over k. The code exposes functions of all scalars, and scalar r, p with vector k.
