/// @file gram_schmidt.hpp
/// @brief Classical Gram-Schmidt orthogonalisation for column-based matrices.
///
/// Given a matrix M whose columns are a set of (possibly non-orthogonal)
/// vectors, the Gram-Schmidt process produces an orthonormal set of vectors
/// that spans the same column space.  Each column is successively projected
/// against the previously orthonormalised columns and the residual is
/// normalised to unit length.
///
/// This is the "classical" Gram-Schmidt variant (CGS).  CGS is simple and
/// efficient but numerically less stable than "modified" Gram-Schmidt (MGS).
/// For the purposes of this library the simplicity is preferred; if higher
/// numerical stability is needed, consider re-orthogonalisation or switching
/// to a Householder-based QR factorisation.
///
/// Two functions are provided:
///
///   1. gram_schmidt_in_place(M) -- orthonormalises the columns of M directly,
///      modifying M in place.  Use this when you no longer need the original
///      matrix and want to avoid an extra copy.
///
///   2. gram_schmidt(M) -- returns a new matrix with orthonormalised columns,
///      leaving the original M unchanged.  This is a convenience wrapper that
///      copies M and then calls the in-place version.  Use this when you need
///      to keep the original matrix intact.
///
/// Both functions handle (near-)linear-dependence gracefully: if a column's
/// residual after projection has squared norm smaller than machine epsilon
/// times its original squared norm, that column is zeroed out rather than
/// normalised.  This avoids amplifying floating-point noise into a spurious
/// unit vector that does not represent a genuine independent direction.

#if !defined(ZIPPER_UTILS_ORTHOGONALIZATION_GRAM_SCHMIDT_HPP)
#define ZIPPER_UTILS_ORTHOGONALIZATION_GRAM_SCHMIDT_HPP

#include <limits>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/utils/detail/dot.hpp>

namespace zipper::utils::orthogonalization {

/// @brief Orthonormalise the columns of M in place using classical
///        Gram-Schmidt.
///
/// After this call, each non-zero column of M is a unit vector, and all
/// columns are mutually orthogonal.  Columns that were (numerically) linearly
/// dependent on earlier columns are set to the zero vector.
///
/// @param[in,out] M  Matrix whose columns are orthonormalised in place.
///
/// Algorithm outline for column j:
///   1. Save the original squared norm of column j (for the dependence check).
///   2. Subtract from column j its projection onto each previously processed
///      column k < j  (each of which is already unit-length).
///   3. If the resulting residual is negligibly small relative to the original
///      norm, the column was (nearly) linearly dependent -- zero it out.
///      Otherwise normalise it to unit length.
template <concepts::Matrix Derived> void gram_schmidt_in_place(Derived &M) {
  using T = typename std::decay_t<Derived>::value_type;
  constexpr T eps = std::numeric_limits<T>::epsilon();
  constexpr index_type RowsStatic =
      std::decay_t<Derived>::extents_type::static_extent(0);
  using Vec = Vector<T, RowsStatic>;

  for (index_type j = 0; j < M.extent(1); ++j) {
    auto u = M.col(j);
    // Record the original energy so we can detect linear dependence.
    Vec vu(u);
    T n = utils::detail::dot(vu, vu);

    for (index_type k = 0; k < j; ++k) {
      auto v = M.col(k);
      // Subtract the component of u along the already-orthonormal v.
      Vec vv(v);
      vu -= utils::detail::dot(vv, vu) * vv;
    }

    if (utils::detail::dot(vu, vu) < eps * n) {
      // The residual is negligible -- this column was (nearly) a linear
      // combination of the previous columns.  Zero it out to avoid
      // normalising numerical noise into a bogus unit vector.
      u = expression::nullary::Constant(T{0}, vu.extents());
    } else {
      u = vu.normalized();
    }
  }
}

/// @brief Return a copy of M with orthonormalised columns.
///
/// This is a non-destructive convenience wrapper around
/// gram_schmidt_in_place().  It copies M into a new matrix, applies the
/// in-place algorithm, and returns the result.  Use this overload when you
/// need to preserve the original matrix (e.g., for later comparison or when M
/// is a const reference).
///
/// @param M  Input matrix (unmodified).
/// @return   A new matrix whose columns are the orthonormalised versions of
///           the columns of M.
template <concepts::Matrix Derived> auto gram_schmidt(const Derived &M) {
  using MType = std::decay_t<Derived>;
  using T = typename MType::value_type;
  constexpr index_type Rows = MType::extents_type::static_extent(0);
  constexpr index_type Cols = MType::extents_type::static_extent(1);

  Matrix<T, Rows, Cols> O(M);
  gram_schmidt_in_place(O);
  return O;
}

} // namespace zipper::utils::orthogonalization

#endif
