/// @file matrix_norm.hpp
/// @brief Matrix norm computations: Frobenius, 1-norm, infinity-norm, spectral.
/// @ingroup decompositions
///
/// Provides four standard matrix norms:
///
///   - **Frobenius norm**: sqrt(sum of squares of all entries).
///     Equivalent to treating the matrix as a flat vector and computing its
///     L2 norm.
///
///   - **1-norm** (max column sum): max over columns of the sum of absolute
///     values in that column.
///
///   - **Infinity norm** (max row sum): max over rows of the sum of absolute
///     values in that row.
///
///   - **Spectral norm** (2-norm): the largest singular value of the matrix.
///     Computed via SVD.
///
/// All functions accept any type satisfying `concepts::Matrix` and return
/// the scalar `value_type` of that matrix.

#if !defined(ZIPPER_UTILS_MATRIX_NORM_HPP)
#define ZIPPER_UTILS_MATRIX_NORM_HPP

#include <algorithm>
#include <cmath>

#include <zipper/concepts/Matrix.hpp>
#include <zipper/utils/decomposition/svd.hpp>

namespace zipper::utils {

/// @brief Frobenius norm: sqrt(sum_{i,j} |A(i,j)|^2).
///
/// @param A  A matrix satisfying `concepts::Matrix`.
/// @return   The Frobenius norm as a scalar.
template <concepts::Matrix Derived>
auto frobenius_norm(const Derived &A) -> typename std::decay_t<Derived>::value_type {
    using T = typename std::decay_t<Derived>::value_type;
    const index_type m = A.extent(0);
    const index_type n = A.extent(1);
    T sum = T{0};
    for (index_type i = 0; i < m; ++i) {
        for (index_type j = 0; j < n; ++j) {
            T v = A(i, j);
            sum += v * v;
        }
    }
    return std::sqrt(sum);
}

/// @brief 1-norm (max column sum): max_j sum_i |A(i,j)|.
///
/// @param A  A matrix satisfying `concepts::Matrix`.
/// @return   The 1-norm as a scalar.
template <concepts::Matrix Derived>
auto one_norm(const Derived &A) -> typename std::decay_t<Derived>::value_type {
    using T = typename std::decay_t<Derived>::value_type;
    const index_type m = A.extent(0);
    const index_type n = A.extent(1);
    T max_col_sum = T{0};
    for (index_type j = 0; j < n; ++j) {
        T col_sum = T{0};
        for (index_type i = 0; i < m; ++i) {
            col_sum += std::abs(A(i, j));
        }
        max_col_sum = std::max(max_col_sum, col_sum);
    }
    return max_col_sum;
}

/// @brief Infinity norm (max row sum): max_i sum_j |A(i,j)|.
///
/// @param A  A matrix satisfying `concepts::Matrix`.
/// @return   The infinity norm as a scalar.
template <concepts::Matrix Derived>
auto inf_norm(const Derived &A) -> typename std::decay_t<Derived>::value_type {
    using T = typename std::decay_t<Derived>::value_type;
    const index_type m = A.extent(0);
    const index_type n = A.extent(1);
    T max_row_sum = T{0};
    for (index_type i = 0; i < m; ++i) {
        T row_sum = T{0};
        for (index_type j = 0; j < n; ++j) {
            row_sum += std::abs(A(i, j));
        }
        max_row_sum = std::max(max_row_sum, row_sum);
    }
    return max_row_sum;
}

/// @brief Spectral norm (2-norm): largest singular value of A.
///
/// Computed via SVD.  For a 1x1 matrix this is just |A(0,0)|.
///
/// @param A  A matrix satisfying `concepts::Matrix`.
/// @return   The spectral norm (largest singular value) as a scalar.
template <concepts::Matrix Derived>
auto spectral_norm(const Derived &A) -> typename std::decay_t<Derived>::value_type {
    using T = typename std::decay_t<Derived>::value_type;
    const index_type m = A.extent(0);
    const index_type n = A.extent(1);

    // Degenerate case: empty matrix.
    if (m == 0 || n == 0) {
        return T{0};
    }

    // 1x1 special case.
    if (m == 1 && n == 1) {
        return std::abs(A(0, 0));
    }

    auto result = decomposition::svd(A);
    // Singular values are in descending order; the first is the largest.
    return result.S(0);
}

} // namespace zipper::utils

#endif
