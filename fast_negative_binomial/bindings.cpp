#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "fast_nb.cpp"

namespace py = pybind11;

PYBIND11_MODULE(fast_negative_binomial, m) {
    m.doc() = "Python bindings for Negative Binomial PMF";

    // Overload all of these

    m.def("negative_binomial",
          [](int k, int r, double p) -> double {
              return nb_base<int>(k, r, p);
          },
          py::arg("k"), py::arg("r"), py::arg("p"),
          "Compute the Negative Binomial PMF.\n\n"
          "Parameters:\n"
          "    k (int): Number of failures.\n"
          "    r (int): Number of successes.\n"
          "    p (float): Probability of success on an individual trial.\n\n"
          "Returns:\n"
          "    float: The PMF value.");

    m.def("negative_binomial",
        [](std::vector<int> k, int r, double p) -> std::vector<double> {
        return nb_base_vec<int>(k, r, p);
    },
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

    m.def("negative_binomial_eigen",
        [](Eigen::VectorXi k, int r, double p) -> Eigen::VectorXd {
        return nb_base_vec_eigen<int>(k, r, p);
    },
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

    // Numpyro calls this the NegativeBinomial2 so we will too.
    // This is in terms of 'mean and concentration (r)', and is generalised
    // such that r is real

    // Overload the vector form of these also.

    m.def("negative_binomial2", py::overload_cast<int, double, double>(&nb2_base),
        py::arg("k"), py::arg("r"), py::arg("p"),
        R"pbdoc(
            Compute the Negative Binomial PMF for a list of k values.

            Parameters:
                k (int): Observation
                r (float): Concentration of distribution
                p (float): Mean of distribution

            Returns:
                float: The PMF values.
        )pbdoc");

    m.def("negative_binomial2", py::overload_cast<std::vector<int>, double, double>(&nb2_base_vec),
        py::arg("k"), py::arg("r"), py::arg("p"),
        R"pbdoc(
            Compute the Negative Binomial PMF for a list of k values.

            Parameters:
                k (List[int]): Observations
                r (float): Concentration of distribution
                p (float): Mean of distribution

            Returns:
                List[float]: The PMF values.
        )pbdoc");
}
