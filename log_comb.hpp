#include <cmath>

double compute_log_comb(int k, int r) {

    if(k < 0 || r <= 0){
        return -std::numeric_limits<double>::infinity();
    }

    return std::lgamma(k + r) - std::lgamma(r) - std::lgamma(k + 1);
}
