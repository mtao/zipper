/// @file pseudoinverse.hpp
/// @brief Moore-Penrose pseudoinverse via SVD.
/// @ingroup decompositions
///
/// Computes the Moore-Penrose pseudoinverse A+ of a matrix A:
///
///   A+ = V * diag(1/sigma_i for sigma_i > tol, else 0) * U^T
///
/// where A = U * diag(S) * V^T is the SVD.  The tolerance determines which
/// singular values are treated as zero (rank deficiency).
///
/// The pseudoinverse satisfies the four Moore-Penrose conditions:
///   1. A  * A+ * A  = A
///   2. A+ * A  * A+ = A+
///   3. (A * A+)^T   = A * A+
///   4. (A+ * A)^T   = A+ * A
///
/// @code
///   auto A = Matrix<double, 3, 2>({{1, 0}, {0, 1}, {0, 0}});
///   auto A_pinv = zipper::utils::pseudoinverse(A);
///   // A_pinv is 2x3, A_pinv * A ≈ I_2
/// @endcode

#if !defined(ZIPPER_UTILS_PSEUDOINVERSE_HPP)
#define ZIPPER_UTILS_PSEUDOINVERSE_HPP

#include <algorithm>
#include <cmath>
#include <limits>

#include <zipper/Matrix.hpp>
#include <zipper/concepts/Matrix.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/utils/decomposition/svd.hpp>

namespace zipper::utils {

/// @brief Compute the Moore-Penrose pseudoinverse of a matrix.
///
/// @param A    An m x n matrix satisfying `concepts::Matrix`.
/// @param tol  Tolerance for treating singular values as zero.  If negative
///             (default), uses max(m,n) * eps * sigma_max.
/// @return     The n x m pseudoinverse matrix.
template <concepts::Matrix Derived>
auto pseudoinverse(const Derived &A,
                   typename std::decay_t<Derived>::value_type tol = -1)
    -> Matrix<typename std::decay_t<Derived>::value_type,
              std::decay_t<Derived>::extents_type::static_extent(1),
              std::decay_t<Derived>::extents_type::static_extent(0)> {
    using AType = std::decay_t<Derived>;
    using T = typename AType::value_type;
    constexpr index_type M = AType::extents_type::static_extent(0);
    constexpr index_type N = AType::extents_type::static_extent(1);

    const index_type m = A.extent(0);
    const index_type n = A.extent(1);

    // Result is n x m (transposed dimensions).
    Matrix<T, N, M> result(n, m);
    result = expression::nullary::Constant(T{0}, result.extents());

    if (m == 0 || n == 0) {
        return result;
    }

    // Compute SVD: A = U * diag(S) * Vt
    auto svd_result = decomposition::svd(A);
    const auto &U = svd_result.U;    // m x p
    const auto &S = svd_result.S;    // p
    const auto &Vt = svd_result.Vt;  // p x n

    const index_type p = std::min(m, n);

    // Determine tolerance.
    if (tol < T{0}) {
        tol = static_cast<T>(std::max(m, n)) *
              std::numeric_limits<T>::epsilon() * S(0);
    }

    // A+ = V * diag(1/sigma_i or 0) * U^T
    //    = Vt^T * diag(...) * U^T
    //
    // Build column by column: result(:, i) = sum_k (1/S_k) * V(:,k) * U(i,k)
    // where V(:,k) = Vt.row(k)^T.
    //
    // Equivalently: result = Vt^T * S_inv_diag * U^T
    // We compute this as:
    //   for each k with S(k) > tol:
    //     result += (1/S(k)) * col_k_of_Vt_T * row_k_of_U_T
    //     i.e.,    result += (1/S(k)) * Vt.row(k)^T * U.col(k)^T
    for (index_type k = 0; k < p; ++k) {
        if (S(k) <= tol) { break; }  // S is sorted descending

        T inv_sk = T{1} / S(k);
        // Outer product: Vt.row(k)^T * U.col(k)^T
        // result(i, j) += inv_sk * Vt(k, i) * U(j, k)
        for (index_type i = 0; i < n; ++i) {
            for (index_type j = 0; j < m; ++j) {
                result(i, j) += inv_sk * Vt(k, i) * U(j, k);
            }
        }
    }

    return result;
}

} // namespace zipper::utils

#endif
