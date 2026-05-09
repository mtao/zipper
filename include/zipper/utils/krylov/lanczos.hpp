/// @file lanczos.hpp
/// @brief Lanczos iteration for computing an orthonormal basis of a Krylov
///        subspace of a symmetric (Hermitian) matrix.
///
/// The Lanczos algorithm is the symmetric specialisation of the Arnoldi
/// iteration (see arnoldi.hpp).  Given a **symmetric** matrix M and an initial
/// vector v1, it builds an orthonormal basis V for the Krylov subspace:
///
///   K_N(M, v1) = span{v1, M v1, M^2 v1, ..., M^{N-1} v1}
///
/// and simultaneously produces a **symmetric tridiagonal** matrix T such that:
///
///   M = V T V^T   (projected onto the Krylov subspace)
///
/// Because M is symmetric, the upper Hessenberg matrix that Arnoldi would
/// produce is actually tridiagonal, so only the diagonal (alpha) and
/// sub-/super-diagonal (beta) entries need to be computed and stored.  This
/// makes each Lanczos step cheaper than the corresponding Arnoldi step: the
/// inner orthogonalisation loop is replaced by a short three-term recurrence,
/// giving O(n) work per step instead of O(n*j).
///
/// The eigenvalues of T approximate extreme eigenvalues of M (Ritz values),
/// making this the backbone of algorithms like the Lanczos eigensolver and
/// the Conjugate Gradient method.
///
/// **Important**: This implementation assumes M is symmetric.  If M is not
/// symmetric, use the general Arnoldi iteration in arnoldi.hpp instead.
///
/// Like arnoldi(), a single public function is provided using `auto n_param`
/// to unify the compile-time and runtime subspace dimension cases:
///
///   - Pass a plain `index_type` for runtime sizing.
///   - Pass `std::integral_constant<index_type, N>{}` (i.e.
///     `static_index_t<N>{}`) to fix the dimension at compile time, allowing
///     fixed-size storage for T (stack allocation, better SIMD vectorisation
///     for small N).

#if !defined(ZIPPER_UTILS_KRYLOV_LANCZOS_HPP)
#define ZIPPER_UTILS_KRYLOV_LANCZOS_HPP

#include <limits>
#include <type_traits>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/detail/assert.hpp>
#include <zipper/detail/is_integral_constant.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/utils/detail/dot.hpp>

namespace zipper::utils::krylov {

/// Result of Lanczos iteration.
/// Contains the orthonormal Lanczos vectors V and symmetric tridiagonal
/// matrix T such that M = V T V^T (restricted to the Krylov subspace).
///
/// Template parameters:
///   T           -- scalar type (e.g. float, double).
///   RowsStatic  -- compile-time row count of the input matrix (or
///                  dynamic_extent if only known at runtime).
///   NStatic     -- compile-time Krylov subspace dimension (or dynamic_extent
///                  for runtime sizing).  When static, V and T use fixed-size
///                  column/row counts, enabling stack allocation.
template <typename T, index_type RowsStatic,
          index_type NStatic = dynamic_extent>
struct LanczosResult {
  static constexpr index_type VCols = NStatic;
  static constexpr index_type TRows = NStatic;
  static constexpr index_type TCols = NStatic;

  /// Orthonormal Lanczos vectors (m x N).  Column j holds v_j.
  Matrix<T, RowsStatic, VCols> V;
  /// Symmetric tridiagonal matrix (N x N).  Diagonal entries are the alpha
  /// coefficients; sub-/super-diagonal entries are the beta coefficients.
  Matrix<T, TRows, TCols> T_mat;
};

/// Lanczos iteration.
///
/// Builds an orthonormal basis V for the Krylov subspace
/// {v1, M*v1, M^2*v1, ..., M^(N-1)*v1} and a symmetric tridiagonal matrix T.
///
/// The three-term recurrence at each step j is:
///
///   w       = M v_j
///   alpha_j = w . v_j               (Rayleigh quotient for this step)
///   w      -= alpha_j v_j + beta_j v_{j-1}
///   beta_{j+1} = ||w||              (norm of residual becomes next
///                                    sub-diagonal entry)
///   v_{j+1} = w / beta_{j+1}
///
/// @param M        A symmetric square matrix.
/// @param v1       Initial vector (should be unit-length or at least non-zero).
/// @param n_param  Subspace dimension -- pass a plain index_type for runtime
///                 sizing, or std::integral_constant<index_type, N>{} (aka
///                 static_index_t<N>{}) to fix the dimension at compile time.
template <concepts::Matrix Derived, concepts::Vector BDerived>
  requires(std::is_same_v<typename std::decay_t<Derived>::value_type,
                          typename std::decay_t<BDerived>::value_type>)
auto lanczos(const Derived &M, const BDerived &v1, auto n_param) {
  using MType = std::decay_t<Derived>;
  using Scalar = typename MType::value_type;

  constexpr index_type RowsStatic = MType::extents_type::static_extent(0);
  if constexpr (!MType::extents_traits::is_static) {
    ZIPPER_ASSERT(M.extent(0) == M.extent(1));
  } else {
    constexpr index_type ColsStatic = MType::extents_type::static_extent(1);
    static_assert(ColsStatic == RowsStatic);
  }

  // Determine whether the subspace dimension is known at compile time.
  using NType = decltype(n_param);
  constexpr index_type NStatic = []() constexpr {
    if constexpr (::zipper::detail::is_integral_constant_v<NType>) {
      return NType::value;
    } else {
      return dynamic_extent;
    }
  }();
  const index_type N = static_cast<index_type>(n_param);

  using Result = LanczosResult<Scalar, RowsStatic, NStatic>;
  using Vec = Vector<Scalar, RowsStatic>;

  const index_type m = M.extent(0);
  Matrix<Scalar, RowsStatic, Result::VCols> V(m, N);
  Matrix<Scalar, Result::TRows, Result::TCols> T(N, N);

  // Zero-initialise via Constant expression (zipper matrices do not have
  // setZero(); dynamic storage is value-initialised but static storage
  // backed by std::array is not).
  V = expression::nullary::Constant(Scalar{0}, V.extents());
  T = expression::nullary::Constant(Scalar{0}, T.extents());

  constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();

  // Set the first Lanczos vector.
  V.col(0) = v1;
  Vec w;

  { // ---------- First iteration (no previous vector to subtract) ----------
    auto v = V.col(0);
    w = M * v;
    // alpha_0 = v_0^T M v_0  (Rayleigh quotient)
    const Scalar a = utils::detail::dot(Vec(w), Vec(v));
    T(0, 0) = a;
    // Remove the component along v_0; there is no v_{-1} term yet.
    w -= a * v;
  }

  for (index_type j = 1; j < N; ++j) {
    // beta_j = ||w|| -- stored symmetrically on both sides of the diagonal
    // to keep T explicitly symmetric, which simplifies later
    // eigen-decomposition.
    const Scalar b = w.norm();
    T(j, j - 1) = b;
    T(j - 1, j) = b;

    // Early termination: if the residual norm is negligible the subspace
    // is invariant under M and no new direction can be added.
    if (b < eps) {
      return Result{
          .V = std::move(V),
          .T_mat = std::move(T),
      };
    }

    auto vjm = V.col(j - 1);
    // Normalise w to get the next basis vector v_j.
    V.col(j) = w / b;
    auto vj = V.col(j);

    // Expand the subspace: w = M v_j.
    w = M * vj;

    // alpha_j = v_j^T M v_j.
    const Scalar a = utils::detail::dot(Vec(w), Vec(vj));
    T(j, j) = a;

    // Three-term recurrence: remove components along v_j and v_{j-1}.
    // In exact arithmetic this is sufficient because M is symmetric, so w
    // is already orthogonal to v_0 ... v_{j-2}.
    w -= a * vj + b * vjm;

    // Full re-orthogonalisation against all previous Lanczos vectors.
    // In floating-point the three-term recurrence loses orthogonality
    // progressively; this pass restores it at the cost of O(m*j) work
    // per step (same asymptotic cost as Arnoldi).  This is critical for
    // eigenvalue solvers that rely on the orthogonality of V.
    for (index_type k = 0; k <= j; ++k) {
      auto vk = V.col(k);
      const Scalar proj = utils::detail::dot(Vec(w), Vec(vk));
      w -= proj * vk;
    }
  }

  return Result{
      .V = std::move(V),
      .T_mat = std::move(T),
  };
}

} // namespace zipper::utils::krylov

#endif
