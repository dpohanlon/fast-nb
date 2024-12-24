# Fast negative binomial distribution

There are a few general optimisations:

* A lookup table for log-gamma for small values (<= 85) of k and n (only on GCC with `constexpr` cmath)
* Stirling's approximation for large (> 10) values of k and r
* An optimisation when r is fixed that pre-computes these values and re-uses them every invocation of log-gamma

These result in divergences with the high-precision version of at most ~1E-5. There are some more extreme approximations that can be made:

* Expansion form of `std::exp`
* Expansion form of `std::log`

This is mostly envisioned for calculations where r and p are scalars and k is a vector of reasonable size (~1000). As such the vectorisation is performed over k. The code exposes functions of all scalars, and scalar r, p with vector k.

Optimisations
-------------

Caching
=======

As k is always an integer, sorting the array a head of time allows us to cache the previous result, to be used if the next value of k is the same. This is particularly useful for the case of single cell RNA sequencing data, where numerous small counts are repeated.

There is also a further optimisation that can be made by using the identity $\log \Gamma (x) = \log \Gamma (x - 1) + \log (x)$. If we know the value of $\log \Gamma (x - 1)$ we can compute $\log \Gamma (x)$ at the cost only of a call to $\log(x)$ rather than the more expensive $\log \Gamma (x)$. To attempt this optimisation as often as possible, we process these in order of increasing k, keeping track of the value of k and the corresponding value of $\log \Gamma (x)$. This has the advantage that we only need to cache two values (per thread), rather than a larger lookup table, and we can also compute repeated values in constant time.

Even in regions where there is a gap between the current k and the previous k, due to the expense of calling log gamma, it may still be faster to make repeated calls to $\log (k - 1)$, $\log (k - 2)$, etc.
