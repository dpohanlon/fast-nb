#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fast_nb.cpp"

namespace py = pybind11;

PYBIND11_MODULE(fast_negative_binomial, m) {
    m.doc() = "Python bindings for Negative Binomial PMF";

    m.def("negative_binomial", &negative_binomial_pmf_base,
          py::arg("k"), py::arg("r"), py::arg("p"),
          "Compute the Negative Binomial PMF.\n\n"
          "Parameters:\n"
          "    k (int): Number of failures.\n"
          "    r (int): Number of successes.\n"
          "    p (float): Probability of success on an individual trial.\n\n"
          "Returns:\n"
          "    float: The PMF value.");

    m.def("negative_binomial_vec", &negative_binomial_pmf_vec,
        py::arg("k"), py::arg("r"), py::arg("p"),
        R"pbdoc(
            Compute the Negative Binomial PMF for a list of k values.

            Parameters:
                k (List[int]): Number of failures for each case.
                r (int): Number of successes.
                p (float): Probability of success on an individual trial.

            Returns:
                List[float]: The PMF values.
        )pbdoc");
}
