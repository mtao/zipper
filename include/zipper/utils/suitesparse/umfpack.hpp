/// @file umfpack.hpp
/// @brief UMFPACK sparse LU factorization bindings.
///
/// Provides sparse LU factorization for general (non-symmetric) square
/// matrices via SuiteSparse UMFPACK.
///
/// Three-tier API (matching zipper's existing solver pattern):
///   - `umfpack_factor(A)`         — factorize, returns UmfpackResult
///   - `UmfpackResult::solve(b)`   — solve using stored factors
///   - `umfpack_solve(A, b)`       — convenience: factor + solve
///
/// The input matrix must be in CSC format (`CSMatrix` with `layout_left`).
/// Convert via `A.as_csc()` if needed.
///
/// @note Only available when `ZIPPER_HAS_SUITESPARSE` is defined.

#if !defined(ZIPPER_UTILS_SUITESPARSE_UMFPACK_HPP)
#define ZIPPER_UTILS_SUITESPARSE_UMFPACK_HPP

#ifdef ZIPPER_HAS_SUITESPARSE

#include <umfpack.h>

#include <expected>
#include <memory>
#include <vector>

#include <zipper/CSMatrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/solver/result.hpp>
#include <zipper/utils/suitesparse/common.hpp>

namespace zipper::utils::suitesparse {

// ═══════════════════════════════════════════════════════════════════════
// RAII wrappers for UMFPACK opaque handles
// ═══════════════════════════════════════════════════════════════════════

namespace detail {

struct UmfpackSymbolicDeleter {
  void operator()(void *p) const {
    if (p) umfpack_dl_free_symbolic(&p);
  }
};

struct UmfpackNumericDeleter {
  void operator()(void *p) const {
    if (p) umfpack_dl_free_numeric(&p);
  }
};

using UmfpackSymbolicPtr = std::unique_ptr<void, UmfpackSymbolicDeleter>;
using UmfpackNumericPtr = std::unique_ptr<void, UmfpackNumericDeleter>;

} // namespace detail

// ═══════════════════════════════════════════════════════════════════════
// UmfpackResult — factorization result with solve capability
// ═══════════════════════════════════════════════════════════════════════

/// Result of a sparse LU factorization via UMFPACK.
///
/// Owns the UMFPACK numeric handle and the CSC arrays needed for solve.
///
/// @tparam N  Static dimension of the matrix (rows), or `dynamic_extent`.
template <index_type N = dynamic_extent> struct UmfpackResult {
  /// UMFPACK numeric factorization handle.
  detail::UmfpackNumericPtr numeric;
  /// Matrix dimensions.
  index_type nrow, ncol;
  /// CSC arrays — owned copies so the factor is self-contained.
  std::vector<SuiteSparse_long> colptr;
  std::vector<SuiteSparse_long> rowind;
  std::vector<double> values;

  /// @brief Solve Ax = b using the stored LU factors.
  ///
  /// @param b  Right-hand side vector.
  /// @return   `std::expected<Vector<double,N>, SolverError>`.
  template <concepts::Vector BDerived>
  auto solve(const BDerived &b) const
      -> std::expected<Vector<double, N>, solver::SolverError> {
    using Result = std::expected<Vector<double, N>, solver::SolverError>;

    // Copy RHS into a contiguous double array.
    std::vector<double> bx(nrow);
    for (index_type i = 0; i < nrow; ++i) {
      bx[i] = b(i);
    }

    std::vector<double> xx(nrow);

    int status = umfpack_dl_solve(
        UMFPACK_A, colptr.data(), rowind.data(), values.data(),
        xx.data(), bx.data(), numeric.get(), nullptr, nullptr);

    if (status != UMFPACK_OK) {
      return Result{std::unexpected(solver::SolverError{
          .kind = solver::SolverError::Kind::breakdown,
          .message = "UMFPACK: solve failed (status " +
                     std::to_string(status) + ")"})};
    }

    Vector<double, N> x(nrow);
    for (index_type i = 0; i < nrow; ++i) {
      x(i) = xx[i];
    }
    return Result{std::move(x)};
  }
};

// ═══════════════════════════════════════════════════════════════════════
// umfpack_factor — sparse LU factorization
// ═══════════════════════════════════════════════════════════════════════

/// @brief Compute the sparse LU factorization of a general matrix
///        in CSC format via UMFPACK.
///
/// @param A  An n x n matrix in CSC layout.
/// @return   `std::expected<UmfpackResult<N>, SolverError>`.
template <typename T, index_type R, index_type C>
auto umfpack_factor(const CSMatrix<T, R, C, storage::layout_left> &A)
    -> std::expected<UmfpackResult<R>, solver::SolverError> {
  static_assert(std::is_same_v<T, double>,
                "UMFPACK only supports double precision");
  constexpr index_type N = R;
  using Result = std::expected<UmfpackResult<N>, solver::SolverError>;

  const auto &cd = A.compressed_data();
  const index_type nrow = A.rows();
  const index_type ncol = A.cols();
  const size_t nnz = cd.nnz();

  // Copy index arrays to SuiteSparse_long (may be a reinterpret on LP64,
  // but we do a proper copy for portability and to make the result
  // self-contained / independent of A's lifetime).
  std::vector<SuiteSparse_long> colptr(ncol + 1);
  for (index_type j = 0; j <= ncol; ++j) {
    colptr[j] = static_cast<SuiteSparse_long>(cd.m_indptr[j]);
  }
  std::vector<SuiteSparse_long> rowind(nnz);
  for (size_t k = 0; k < nnz; ++k) {
    rowind[k] = static_cast<SuiteSparse_long>(cd.m_indices[k]);
  }
  std::vector<double> vals(cd.m_values.begin(), cd.m_values.end());

  // Symbolic analysis.
  void *symbolic_raw = nullptr;
  int status = umfpack_dl_symbolic(
      static_cast<SuiteSparse_long>(nrow),
      static_cast<SuiteSparse_long>(ncol),
      colptr.data(), rowind.data(), vals.data(),
      &symbolic_raw, nullptr, nullptr);
  detail::UmfpackSymbolicPtr symbolic(symbolic_raw);

  if (status != UMFPACK_OK) {
    return Result{std::unexpected(solver::SolverError{
        .kind = solver::SolverError::Kind::breakdown,
        .message = "UMFPACK: symbolic analysis failed (status " +
                   std::to_string(status) + ")"})};
  }

  // Numeric factorization.
  void *numeric_raw = nullptr;
  status = umfpack_dl_numeric(
      colptr.data(), rowind.data(), vals.data(),
      symbolic.get(), &numeric_raw, nullptr, nullptr);
  detail::UmfpackNumericPtr numeric(numeric_raw);

  if (status != UMFPACK_OK) {
    return Result{std::unexpected(solver::SolverError{
        .kind = solver::SolverError::Kind::breakdown,
        .message = "UMFPACK: numeric factorization failed (status " +
                   std::to_string(status) + ")"})};
  }

  return Result{UmfpackResult<N>{
      .numeric = std::move(numeric),
      .nrow = nrow,
      .ncol = ncol,
      .colptr = std::move(colptr),
      .rowind = std::move(rowind),
      .values = std::move(vals),
  }};
}

// ═══════════════════════════════════════════════════════════════════════
// umfpack_solve — convenience: factor + solve
// ═══════════════════════════════════════════════════════════════════════

/// @brief Solve Ax = b via sparse LU (UMFPACK).
///
/// @param A  An n x n matrix in CSC layout.
/// @param b  Right-hand side vector of length n.
/// @return   `std::expected<Vector<double,N>, SolverError>`.
template <typename T, index_type R, index_type C, concepts::Vector BDerived>
auto umfpack_solve(const CSMatrix<T, R, C, storage::layout_left> &A,
                   const BDerived &b)
    -> std::expected<Vector<double, R>, solver::SolverError> {
  auto result = umfpack_factor(A);
  if (!result) {
    return std::unexpected(std::move(result.error()));
  }
  return result->solve(b);
}

} // namespace zipper::utils::suitesparse

#endif // ZIPPER_HAS_SUITESPARSE
#endif // ZIPPER_UTILS_SUITESPARSE_UMFPACK_HPP
