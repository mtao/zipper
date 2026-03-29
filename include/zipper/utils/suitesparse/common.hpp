/// @file common.hpp
/// @brief RAII wrappers and conversion helpers for SuiteSparse integration.
///
/// Provides:
///   - `CholmodCommon`: RAII wrapper around `cholmod_common` (start/finish).
///   - `to_cholmod_sparse()`: zero-copy conversion from CSMatrix (CSC) to
///     `cholmod_sparse` view (non-owning).
///   - `to_cholmod_dense()`: conversion from zipper Vector/Matrix to
///     `cholmod_dense` (copy into CHOLMOD-allocated storage).
///   - `from_cholmod_dense()`: conversion from `cholmod_dense` back to
///     zipper Vector.
///
/// All SuiteSparse types use `SuiteSparse_long` (= `int64_t`) for indices.
/// Zipper uses `size_t` (= `uint64_t` on LP64).  On LP64 platforms these
/// have the same width, so index arrays can be reinterpret_cast'd.  A
/// static_assert guards this assumption.
///
/// @note This header is only available when `ZIPPER_HAS_SUITESPARSE` is
///       defined (set by the build system when SuiteSparse is found).

#if !defined(ZIPPER_UTILS_SUITESPARSE_COMMON_HPP)
#define ZIPPER_UTILS_SUITESPARSE_COMMON_HPP

#ifdef ZIPPER_HAS_SUITESPARSE

#include <cholmod.h>

#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>

#include <zipper/CSMatrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/types.hpp>

namespace zipper::utils::suitesparse {

// ── Index width guard ─────────────────────────────────────────────────
// SuiteSparse uses SuiteSparse_long (int64_t) for indices.
// Zipper uses size_t.  On LP64 platforms both are 8 bytes, so we can
// reinterpret_cast index arrays instead of copying.
static_assert(sizeof(index_type) == sizeof(SuiteSparse_long),
              "zipper::index_type and SuiteSparse_long must have the same "
              "width for zero-copy index interop");

// ── Dimension safety check ────────────────────────────────────────────
// SuiteSparse_long is signed (int64_t).  Zipper's index_type is unsigned
// (size_t).  Although both are 8 bytes on LP64, a size_t value exceeding
// INT64_MAX would be misinterpreted as negative by SuiteSparse.
//
// Rather than checking every index element, we validate that the matrix
// dimensions and nnz fit in SuiteSparse_long.  Since all index values
// are bounded by max(rows, cols) and all indptr values by nnz, this
// single upfront check guarantees every element is in range.
constexpr auto max_suitesparse_index =
    static_cast<index_type>(std::numeric_limits<SuiteSparse_long>::max());

/// Verify that matrix dimensions and nnz are representable as
/// SuiteSparse_long.  Throws std::overflow_error if not.
inline void check_suitesparse_bounds(index_type nrow, index_type ncol,
                                     size_t nnz) {
  if (nrow > max_suitesparse_index || ncol > max_suitesparse_index ||
      nnz > max_suitesparse_index) {
    throw std::overflow_error(
        "Matrix dimensions or nnz exceed SuiteSparse_long range "
        "(INT64_MAX). Cannot safely convert size_t indices to "
        "SuiteSparse_long.");
  }
}

// ═══════════════════════════════════════════════════════════════════════
// CholmodCommon — RAII wrapper around cholmod_common
// ═══════════════════════════════════════════════════════════════════════

/// RAII wrapper that calls `cholmod_l_start` on construction and
/// `cholmod_l_finish` on destruction.  All CHOLMOD operations require
/// a `cholmod_common*`, so this must outlive any CHOLMOD objects.
class CholmodCommon {
public:
  CholmodCommon() { cholmod_l_start(&m_common); }
  ~CholmodCommon() { cholmod_l_finish(&m_common); }

  CholmodCommon(const CholmodCommon &) = delete;
  CholmodCommon &operator=(const CholmodCommon &) = delete;
  CholmodCommon(CholmodCommon &&) = delete;
  CholmodCommon &operator=(CholmodCommon &&) = delete;

  auto get() -> cholmod_common * { return &m_common; }
  auto get() const -> const cholmod_common * { return &m_common; }

private:
  cholmod_common m_common{};
};

// ═══════════════════════════════════════════════════════════════════════
// to_cholmod_sparse — zero-copy CSC matrix view
// ═══════════════════════════════════════════════════════════════════════

/// Create a non-owning `cholmod_sparse` view of a CSMatrix in CSC layout.
///
/// The returned `cholmod_sparse` points directly into the CSMatrix's
/// internal arrays (indptr, indices, values).  The CSMatrix must outlive
/// the returned view.
///
/// @tparam T   Scalar type (must be `double` — CHOLMOD only supports real
///             double precision in its standard configuration).
/// @tparam R   Row extent (static or dynamic).
/// @tparam C   Column extent (static or dynamic).
///
/// @param A    A CSC-format sparse matrix (`CSMatrix` with `layout_left`).
/// @return     A stack-allocated `cholmod_sparse` (non-owning view).
///
/// @note The caller must NOT call `cholmod_l_free_sparse` on the result,
///       since it does not own the underlying memory.
template <typename T, index_type R, index_type C>
auto to_cholmod_sparse(const CSMatrix<T, R, C, storage::layout_left> &A)
    -> cholmod_sparse {
  static_assert(std::is_same_v<T, double>,
                "CHOLMOD only supports double precision");

  const auto &cd = A.compressed_data();

  // Verify dimensions fit in SuiteSparse_long before reinterpreting
  // index arrays from size_t to int64_t.
  check_suitesparse_bounds(A.rows(), A.cols(), cd.nnz());

  cholmod_sparse cs{};
  cs.nrow = static_cast<size_t>(A.rows());
  cs.ncol = static_cast<size_t>(A.cols());
  cs.nzmax = cd.nnz();

  // Column pointers (outer pointer array for CSC).
  // CHOLMOD expects SuiteSparse_long*, we have index_type* — same width
  // on LP64 (guarded by static_assert above).
  cs.p = const_cast<void *>(
      static_cast<const void *>(cd.indptr_data()));

  // Row indices (inner index array for CSC).
  cs.i = const_cast<void *>(
      static_cast<const void *>(cd.indices_data()));

  // Values.
  cs.x = const_cast<void *>(
      static_cast<const void *>(cd.values_data()));

  cs.z = nullptr; // No imaginary part (real matrix).

  cs.stype = 0;   // 0 = unsymmetric (both triangles stored).
                   // Caller may override for symmetric matrices.
  cs.itype = CHOLMOD_LONG; // SuiteSparse_long indices.
  cs.xtype = CHOLMOD_REAL; // Real values.
  cs.dtype = CHOLMOD_DOUBLE;
  cs.sorted = 1;  // Indices within each column are sorted.
  cs.packed = 1;  // No gaps between columns (standard CSC).

  return cs;
}

/// Overload for symmetric matrices.  Sets `stype = 1` (upper triangle
/// stored) or `stype = -1` (lower triangle stored).
///
/// @param A      A CSC-format sparse matrix.
/// @param stype  1 = upper triangle stored, -1 = lower triangle stored.
template <typename T, index_type R, index_type C>
auto to_cholmod_sparse(const CSMatrix<T, R, C, storage::layout_left> &A,
                       int stype) -> cholmod_sparse {
  auto cs = to_cholmod_sparse(A);
  cs.stype = stype;
  return cs;
}

// ═══════════════════════════════════════════════════════════════════════
// to_cholmod_dense — copy Vector to cholmod_dense
// ═══════════════════════════════════════════════════════════════════════

/// Convert a zipper Vector to a CHOLMOD-owned dense column vector.
///
/// Allocates a `cholmod_dense` via `cholmod_l_allocate_dense` and copies
/// the vector elements using zipper's expression interface.
///
/// @param b   Input vector.
/// @param cc  CHOLMOD common struct (for allocation).
/// @return    Pointer to newly allocated `cholmod_dense`.
template <concepts::Vector BDerived>
auto to_cholmod_dense(const BDerived &b, cholmod_common *cc)
    -> cholmod_dense * {
  using T = typename std::decay_t<BDerived>::value_type;
  static_assert(std::is_same_v<std::remove_const_t<T>, double>,
                "CHOLMOD only supports double precision");

  const index_type n = b.rows();
  cholmod_dense *B =
      cholmod_l_allocate_dense(n, 1, n, CHOLMOD_REAL, cc);
  if (!B) {
    throw std::runtime_error("cholmod_l_allocate_dense failed");
  }

  // Wrap CHOLMOD's buffer as a zipper span view and assign from b.
  auto dst = VectorBase(std::span<double>(static_cast<double *>(B->x), n));
  dst = b;

  return B;
}

// ═══════════════════════════════════════════════════════════════════════
// from_cholmod_dense — extract Vector from cholmod_dense
// ═══════════════════════════════════════════════════════════════════════

/// Convert a `cholmod_dense` column vector to a zipper Vector.
///
/// Wraps the CHOLMOD buffer as a non-owning zipper span view and copies
/// into an owned Vector via zipper's assignment infrastructure.
///
/// @tparam N   Static extent for the result vector (use `dynamic_extent`
///             for runtime-sized).
/// @param X    CHOLMOD dense matrix (must be n x 1).
/// @return     A new zipper Vector owning the data.
template <index_type N = dynamic_extent>
auto from_cholmod_dense(const cholmod_dense *X) -> Vector<double, N> {
  const index_type n = X->nrow;
  // Wrap CHOLMOD's buffer as a const zipper span view.
  auto src = VectorBase(
      std::span<const double>(static_cast<const double *>(X->x), n));
  // Construct owned Vector from the span view (deep copy via assign).
  Vector<double, N> result(src);
  return result;
}

// ═══════════════════════════════════════════════════════════════════════
// RAII helper for cholmod_dense
// ═══════════════════════════════════════════════════════════════════════

/// Unique-pointer-style wrapper for `cholmod_dense*` that calls
/// `cholmod_l_free_dense` on destruction.
struct CholmodDenseDeleter {
  cholmod_common *cc;
  void operator()(cholmod_dense *p) const {
    if (p) {
      cholmod_l_free_dense(&p, cc);
    }
  }
};
using CholmodDensePtr = std::unique_ptr<cholmod_dense, CholmodDenseDeleter>;

/// Wrap a raw `cholmod_dense*` in a RAII handle.
inline auto make_cholmod_dense_ptr(cholmod_dense *p, cholmod_common *cc)
    -> CholmodDensePtr {
  return CholmodDensePtr(p, CholmodDenseDeleter{cc});
}

// ═══════════════════════════════════════════════════════════════════════
// RAII helper for cholmod_sparse
// ═══════════════════════════════════════════════════════════════════════

struct CholmodSparseDeleter {
  cholmod_common *cc;
  void operator()(cholmod_sparse *p) const {
    if (p) {
      cholmod_l_free_sparse(&p, cc);
    }
  }
};
using CholmodSparsePtr =
    std::unique_ptr<cholmod_sparse, CholmodSparseDeleter>;

inline auto make_cholmod_sparse_ptr(cholmod_sparse *p, cholmod_common *cc)
    -> CholmodSparsePtr {
  return CholmodSparsePtr(p, CholmodSparseDeleter{cc});
}

// ═══════════════════════════════════════════════════════════════════════
// from_cholmod_sparse — convert cholmod_sparse to CSMatrix (CSC)
// ═══════════════════════════════════════════════════════════════════════

/// Convert a `cholmod_sparse` matrix (CSC format) to a zipper CSMatrix.
///
/// Creates an owned CSMatrix by copying the data from the CHOLMOD sparse
/// matrix via the COO intermediary.  The CHOLMOD matrix must be in
/// packed, sorted CSC format with `xtype == CHOLMOD_REAL` and
/// `dtype == CHOLMOD_DOUBLE`.
///
/// @tparam R  Static row extent (or `dynamic_extent`).
/// @tparam C  Static column extent (or `dynamic_extent`).
/// @param  S  CHOLMOD sparse matrix.
/// @return    An owned CSMatrix in CSC layout.
template <index_type R = dynamic_extent, index_type C = dynamic_extent>
auto from_cholmod_sparse(const cholmod_sparse *S)
    -> CSMatrix<double, R, C, storage::layout_left> {
  const index_type nrow = static_cast<index_type>(S->nrow);
  const index_type ncol = static_cast<index_type>(S->ncol);

  const auto *colptr =
      static_cast<const SuiteSparse_long *>(S->p);
  const auto *rowind =
      static_cast<const SuiteSparse_long *>(S->i);
  const auto *vals = static_cast<const double *>(S->x);

  // Build via COO intermediary.
  COOMatrix<double, R, C> coo(nrow, ncol);
  for (index_type j = 0; j < ncol; ++j) {
    for (SuiteSparse_long k = colptr[j]; k < colptr[j + 1]; ++k) {
      coo.emplace(static_cast<index_type>(rowind[k]), j) = vals[k];
    }
  }
  coo.compress();
  return CSMatrix<double, R, C, storage::layout_left>(coo);
}

} // namespace zipper::utils::suitesparse

#endif // ZIPPER_HAS_SUITESPARSE
#endif // ZIPPER_UTILS_SUITESPARSE_COMMON_HPP
