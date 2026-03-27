/// @file ssor_preconditioner.hpp
/// @brief Symmetric Successive Over-Relaxation (SSOR) preconditioner.
///
/// The SSOR preconditioner is defined as:
///
///   M = (D + omega * L) * D^{-1} * (D + omega * U)  /  (2 - omega)
///
/// where D is the diagonal of A, L is the strict lower triangle, U is the
/// strict upper triangle, and omega is the relaxation parameter (typically
/// in (0, 2), with omega = 1 being the symmetric Gauss-Seidel preconditioner).
///
/// Applying M^{-1} to a vector r is done in three steps:
///   1. Forward sweep:  solve (D + omega * L) * y = r
///   2. Diagonal scale: z = D * y
///   3. Backward sweep: solve (D + omega * U) * w = z
///   4. Scale:          result = (2 - omega) * w
///
/// Rather than forming M explicitly, the apply method performs the sweeps
/// directly using coefficient access on the original matrix.  This stores
/// only a const reference to A and the omega parameter -- no matrix copies.
///
/// @see zipper::concepts::Preconditioner
/// @see zipper::utils::solver::preconditioned_conjugate_gradient

#pragma once

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/as.hpp>

namespace zipper::utils::solver {

/// @brief SSOR preconditioner.
///
/// Stores a const reference to the coefficient matrix and the relaxation
/// parameter omega.  The `.apply(r)` method performs forward/backward sweeps
/// to compute M^{-1} * r without forming M explicitly.
///
/// @tparam T   Scalar type.
/// @tparam Dim Compile-time dimension, or `dynamic_extent`.
/// @tparam MatType The matrix type (deduced from constructor).
template <typename T, index_type Dim = dynamic_extent>
struct SSORPreconditioner {
    using value_type = T;

    /// Pre-computed inverse diagonal: inv_diag(i) = 1 / A(i,i).
    Vector<T, Dim> inv_diag;
    /// Pre-computed diagonal: diag(i) = A(i,i).
    Vector<T, Dim> diag;
    /// Relaxation parameter (0 < omega < 2).
    T omega;
    /// Const reference to the original matrix (for off-diagonal access).
    /// We store a full copy since storing a reference to an arbitrary
    /// expression type would require a template parameter.
    Matrix<T, Dim, Dim> A_copy;

    /// @brief Construct from a matrix and relaxation parameter.
    ///
    /// @param A      The coefficient matrix (SPD with non-zero diagonal).
    /// @param omega  Relaxation parameter (default: 1.0 = symmetric
    /// Gauss-Seidel).
    template <concepts::Matrix MDerived>
    explicit SSORPreconditioner(const MDerived &A, T omega_ = T{1})
      : inv_diag(A.rows()), diag(A.rows()), omega(omega_), A_copy(A) {
        const index_type n = A.rows();
        for (index_type i = 0; i < n; ++i) {
            diag(i) = A(i, i);
            inv_diag(i) = T{1} / A(i, i);
        }
    }

    /// @brief Apply the SSOR preconditioner: z = M^{-1} * r.
    ///
    /// Performs:
    ///   1. Forward sweep:   (D + omega*L) y = r
    ///   2. Diagonal scale:  z_i = D_i * y_i
    ///   3. Backward sweep:  (D + omega*U) w = z
    ///   4. Scale by (2 - omega)
    ///
    /// @param r  Input vector (residual).
    /// @return   Preconditioned vector M^{-1} r.
    template <concepts::Vector VDerived>
    auto apply(const VDerived &r) const -> Vector<T, Dim> {
        const index_type n = diag.extent(0);

        // Step 1: Forward sweep -- solve (D + omega*L) y = r
        // y_i = (r_i - omega * sum_{j<i} A_{ij} * y_j) / A_{ii}
        Vector<T, Dim> y(n);
        for (index_type i = 0; i < n; ++i) {
            T sum = T{0};
            for (index_type j = 0; j < i; ++j) { sum += A_copy(i, j) * y(j); }
            y(i) = (r(i) - omega * sum) * inv_diag(i);
        }

        // Step 2: Diagonal scale -- z_i = D_i * y_i
        auto diag_y = diag.as_array() * y.as_array();
        Vector<T, Dim> z(as_vector(diag_y));

        // Step 3: Backward sweep -- solve (D + omega*U) w = z
        // w_i = (z_i - omega * sum_{j>i} A_{ij} * w_j) / A_{ii}
        Vector<T, Dim> w(n);
        for (index_type i = n; i-- > 0;) {
            T sum = T{0};
            for (index_type j = i + 1; j < n; ++j) {
                sum += A_copy(i, j) * w(j);
            }
            w(i) = (z(i) - omega * sum) * inv_diag(i);
        }

        // Step 4: Scale by (2 - omega)
        w = ((T{2} - omega) * w).eval();

        return w;
    }
};

} // namespace zipper::utils::solver
