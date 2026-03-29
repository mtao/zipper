#if !defined(ZIPPER_CSRMATRIX_HPP)
#define ZIPPER_CSRMATRIX_HPP

/// @file CSRMatrix.hpp
/// @brief Backward-compatible alias for CSR sparse matrices.
/// @ingroup user_types
///
/// `CSRMatrix<T, Rows, Cols>` is a deprecated alias for
/// `CSMatrix<T, Rows, Cols, layout_right>`.  All new code should use
/// `CSMatrix` directly.

#include "CSMatrix.hpp"

namespace zipper {

/// Backward-compatible alias: CSRMatrix is CSMatrix with layout_right (CSR).
template <typename ValueType, index_type Rows, index_type Cols>
using CSRMatrix = CSMatrix<ValueType, Rows, Cols, storage::layout_right>;

// ── COOMatrix::to_csr() definition (needs CSMatrix to be complete) ────
template <typename ValueType, index_type Rows, index_type Cols>
auto COOMatrix<ValueType, Rows, Cols>::to_csr() const
    -> CSMatrix<ValueType, Rows, Cols, storage::layout_right> {
  return CSMatrix<ValueType, Rows, Cols, storage::layout_right>(*this);
}

} // namespace zipper

#endif
