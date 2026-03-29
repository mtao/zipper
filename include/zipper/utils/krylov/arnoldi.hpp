/// @file arnoldi.hpp
/// @brief Arnoldi iteration for computing an orthonormal basis of a Krylov
///        subspace of a general (non-symmetric) square matrix.
///
/// The Arnoldi iteration takes a square matrix M and an initial vector b, and
/// builds an orthonormal basis Q for the Krylov subspace:
///
///   K_N(M, b) = span{b, Mb, M^2 b, ..., M^{N-1} b}
///
/// It simultaneously produces an upper Hessenberg matrix H such that:
///
///   M * Q_N = Q_{N+1} * H_{N+1,N}   (or exactly, if the subspace is invariant)
///
/// This is the foundation for iterative eigenvalue solvers (e.g., implicitly
/// restarted Arnoldi / ARPACK) and Krylov-based linear solvers (e.g., GMRES).
/// Unlike the Lanczos algorithm (see lanczos.hpp), Arnoldi works for general
/// (non-symmetric) matrices -- H is upper Hessenberg rather than tridiagonal.
///
/// A single public function is provided, using `auto n_param` to unify the
/// compile-time and runtime subspace dimension cases:
///
///   - Pass a plain `index_type` for runtime sizing.
///   - Pass `std::integral_constant<index_type, N>{}` (i.e.
///     `static_index_t<N>{}`) to fix the dimension at compile time, allowing
///     fixed-size storage for H (stack allocation, better SIMD vectorisation
///     for small N).

#if !defined(ZIPPER_UTILS_KRYLOV_ARNOLDI_HPP)
#define ZIPPER_UTILS_KRYLOV_ARNOLDI_HPP

#include <limits>
#include <type_traits>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/detail/assert.hpp>
#include <zipper/detail/is_integral_constant.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/utils/detail/dot.hpp>

namespace zipper::utils::krylov {

/// Result of Arnoldi iteration.
/// Contains the orthonormal basis Q and upper Hessenberg matrix H such that
/// M * Q(:, 0:N) = Q * H.
///
/// Template parameters:
///   T           -- scalar type (e.g. float, double).
///   RowsStatic  -- compile-time row count of the input matrix (or
///                  dynamic_extent if only known at runtime).
///   NStatic     -- compile-time Krylov subspace dimension (or dynamic_extent
///                  for runtime sizing).  When static, Q and H use fixed-size
///                  column/row counts, enabling stack allocation.
template <typename T, index_type RowsStatic,
          index_type NStatic = dynamic_extent>
struct ArnoldiResult {
  static constexpr index_type QCols =
      (NStatic == dynamic_extent) ? dynamic_extent : NStatic + 1;
  static constexpr index_type HRows = QCols;
  static constexpr index_type HCols = NStatic;

  /// Orthonormal basis for the Krylov subspace (m x (N+1) columns).
  Matrix<T, RowsStatic, QCols> Q;
  /// Upper Hessenberg matrix ((N+1) x N).
  Matrix<T, HRows, HCols> H;
};

/// Arnoldi iteration.
///
/// Builds an orthonormal basis Q for the Krylov subspace
/// {b, M*b, M^2*b, ..., M^(N-1)*b} and an upper Hessenberg matrix H
/// such that M * Q_N = Q_{N+1} * H.
///
/// @param M  A square matrix.
/// @param b  Initial vector (will be normalised internally).
/// @param n_param  Subspace dimension -- pass a plain index_type for runtime
///                 sizing, or std::integral_constant<index_type, N>{} (aka
///                 static_index_t<N>{}) to fix the dimension at compile time.
template <concepts::Matrix Derived, concepts::Vector BDerived>
  requires(std::is_same_v<typename std::decay_t<Derived>::value_type,
                          typename std::decay_t<BDerived>::value_type>)
auto arnoldi(const Derived &M, const BDerived &b, auto n_param) {
  using MType = std::decay_t<Derived>;
  using T = typename MType::value_type;

  constexpr index_type RowsStatic = MType::extents_type::static_extent(0);
  if constexpr (!MType::extents_traits::is_static) {
    ZIPPER_ASSERT(M.extent(0) == M.extent(1));
  } else {
    constexpr index_type ColsStatic = MType::extents_type::static_extent(1);
    static_assert(ColsStatic == RowsStatic);
  }

  // Determine whether the subspace dimension is known at compile time.
  // If n_param is an integral_constant, NStatic captures the value for
  // fixed-size matrix types; otherwise we fall back to dynamic_extent.
  using NType = decltype(n_param);
  constexpr index_type NStatic = []() constexpr {
    if constexpr (::zipper::detail::is_integral_constant_v<NType>) {
      return NType::value;
    } else {
      return dynamic_extent;
    }
  }();
  const index_type N = static_cast<index_type>(n_param);

  using Result = ArnoldiResult<T, RowsStatic, NStatic>;
  using Vec = Vector<T, RowsStatic>;

  const index_type m = M.extent(0);
  Matrix<T, RowsStatic, Result::QCols> Q(m, N + 1);
  Matrix<T, Result::HRows, Result::HCols> H(N + 1, N);

  // Zero-initialise via Constant expression (zipper matrices do not have
  // setZero(); dynamic storage is value-initialised but static storage
  // backed by std::array is not).
  Q = expression::nullary::Constant(T{0}, Q.extents());
  H = expression::nullary::Constant(T{0}, H.extents());

  constexpr T eps = std::numeric_limits<T>::epsilon();
  Vec v;

  // Normalise the initial vector to form the first basis vector.
  Q.col(0) = b.normalized();

  for (index_type j = 0; j < N; ++j) {
    // Expand the Krylov subspace: v = M * q_j.
    auto q = Q.col(j);
    v = M * q;

    // Orthogonalise v against all existing basis vectors (classical
    // Gram-Schmidt).  The projection coefficients are stored in column j
    // of H, building the upper Hessenberg structure.
    for (index_type k = 0; k <= j; ++k) {
      auto l = Q.col(k);
      const T s = utils::detail::dot(l, v);
      H(k, j) = s;
      v -= s * l;
    }

    // The sub-diagonal entry H(j+1, j) is the norm of the residual.
    const T vnorm = v.norm();
    H(j + 1, j) = vnorm;

    // If the residual is negligibly small the Krylov subspace is
    // (numerically) invariant -- we have found an exact eigenspace and
    // there is no new direction to add, so we stop early.
    if (vnorm <= eps) {
      return Result{
          .Q = std::move(Q),
          .H = std::move(H),
      };
    }

    // Normalise the residual to get the next basis vector.
    Q.col(j + 1) = v / vnorm;
  }

  return Result{
      .Q = std::move(Q),
      .H = std::move(H),
  };
}

} // namespace zipper::utils::krylov

#endif
