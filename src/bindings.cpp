#include <pybind11/pybind11.h>
#include "fast_nb.cpp"

namespace py = pybind11;

PYBIND11_MODULE(negative_binomial, m) {
    m.doc() = "Python bindings for Negative Binomial PMF"; // Optional module docstring

    m.def("negative_binomial", &negative_binomial_pmf_base,
          py::arg("k"), py::arg("r"), py::arg("p"),
          "Compute the Negative Binomial PMF.\n\n"
          "Parameters:\n"
          "    k (int): Number of failures.\n"
          "    r (int): Number of successes.\n"
          "    p (float): Probability of success on an individual trial.\n\n"
          "Returns:\n"
          "    float: The PMF value.");
}
