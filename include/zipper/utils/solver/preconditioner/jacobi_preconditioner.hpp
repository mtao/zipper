/// @file jacobi_preconditioner.hpp
/// @brief Jacobi (diagonal) preconditioner for iterative solvers.
///
/// The Jacobi preconditioner approximates the inverse of A by the inverse of
/// its diagonal: M^{-1} = diag(1/A_{11}, 1/A_{22}, ..., 1/A_{nn}).
///
/// Applying the preconditioner to a vector r computes:
///   z_i = r_i / A_{ii}
///
/// This is the simplest possible preconditioner.  It is effective when A is
/// diagonally dominant, clustering eigenvalues of M^{-1}A near 1.
///
/// The preconditioner pre-computes and stores the inverse diagonal as a
/// Vector (O(n) storage), which is the idiomatic zipper approach: lightweight,
/// no full matrix copy.
///
/// @see zipper::concepts::Preconditioner
/// @see zipper::utils::solver::preconditioned_conjugate_gradient

#if !defined(ZIPPER_UTILS_SOLVER_PRECONDITIONER_JACOBI_PRECONDITIONER_HPP)
#define ZIPPER_UTILS_SOLVER_PRECONDITIONER_JACOBI_PRECONDITIONER_HPP

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/as.hpp>

namespace zipper::utils::solver {

/// @brief Jacobi (diagonal) preconditioner.
///
/// Stores the inverse diagonal of the coefficient matrix and applies it
/// component-wise to a vector.
///
/// @tparam T   Scalar type (e.g. `double`).
/// @tparam Dim Compile-time dimension, or `dynamic_extent`.
template <typename T, index_type Dim = dynamic_extent>
struct JacobiPreconditioner {
    using value_type = T;

    /// Pre-computed inverse diagonal: inv_diag(i) = 1 / A(i,i).
    Vector<T, Dim> inv_diag;

    /// @brief Construct from a matrix by extracting and inverting its diagonal.
    ///
    /// @param A  The coefficient matrix.  Must have non-zero diagonal entries.
    template <concepts::Matrix MDerived>
    explicit JacobiPreconditioner(const MDerived &A) : inv_diag(A.rows()) {
        const index_type n = A.rows();
        for (index_type i = 0; i < n; ++i) { inv_diag(i) = T{1} / A(i, i); }
    }

    /// @brief Apply the preconditioner: z = M^{-1} * r.
    ///
    /// Computes z_i = r_i * inv_diag_i (component-wise multiplication).
    ///
    /// @param r  Input vector.
    /// @return   Preconditioned vector z = M^{-1} r.
    template <concepts::Vector VDerived>
    auto apply(const VDerived &r) const -> Vector<T, Dim> {
        auto arr = inv_diag.as_array() * r.as_array();
        return Vector<T, Dim>(as_vector(arr));
    }
};

} // namespace zipper::utils::solver
#endif
