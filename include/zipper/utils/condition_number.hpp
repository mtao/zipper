/// @file condition_number.hpp
/// @brief Condition number estimation via SVD.
/// @ingroup decompositions
///
/// Computes the 2-norm condition number of a matrix:
///
///   cond(A) = sigma_max / sigma_min
///
/// where sigma_max and sigma_min are the largest and smallest singular values
/// of A, respectively.  If sigma_min is zero (singular matrix), the condition
/// number is infinity.
///
/// @code
///   auto A = Matrix<double, 3, 3>({{1, 0, 0}, {0, 2, 0}, {0, 0, 3}});
///   double kappa = zipper::utils::condition_number(A);
///   // kappa == 3.0 / 1.0 == 3.0
/// @endcode

#if !defined(ZIPPER_UTILS_CONDITION_NUMBER_HPP)
#define ZIPPER_UTILS_CONDITION_NUMBER_HPP

#include <cmath>
#include <limits>

#include <zipper/concepts/Matrix.hpp>
#include <zipper/utils/decomposition/svd.hpp>

namespace zipper::utils {

/// @brief Compute the 2-norm condition number of a matrix.
///
/// @param A  A matrix satisfying `concepts::Matrix`.
/// @return   sigma_max / sigma_min.  Returns infinity if the matrix is
///           singular (sigma_min == 0).
template <concepts::Matrix Derived>
auto condition_number(const Derived &A) -> typename std::decay_t<Derived>::value_type {
    using T = typename std::decay_t<Derived>::value_type;
    const index_type m = A.extent(0);
    const index_type n = A.extent(1);

    if (m == 0 || n == 0) {
        return T{0};
    }

    auto result = decomposition::svd(A);
    const index_type p = std::min(m, n);

    // Singular values are in descending order.
    T sigma_max = result.S(0);
    T sigma_min = result.S(p - 1);

    if (sigma_min <= std::numeric_limits<T>::min()) {
        return std::numeric_limits<T>::infinity();
    }

    return sigma_max / sigma_min;
}

} // namespace zipper::utils

#endif
