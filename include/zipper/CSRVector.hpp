#if !defined(ZIPPER_CSRVECTOR_HPP)
#define ZIPPER_CSRVECTOR_HPP

/// @file CSRVector.hpp
/// @brief Backward-compatible alias for compressed sparse vectors.
/// @ingroup user_types
///
/// `CSRVector<T, N>` is a deprecated alias for `CSVector<T, N>`.
/// All new code should use `CSVector` directly.

#include "CSVector.hpp"

namespace zipper {

/// Backward-compatible alias: CSRVector is CSVector.
template <typename ValueType, index_type N>
using CSRVector = CSVector<ValueType, N>;

// ── COOVector::to_csr() definition (needs CSVector to be complete) ────
template <typename ValueType, index_type N>
auto COOVector<ValueType, N>::to_csr() const -> CSVector<ValueType, N> {
  return CSVector<ValueType, N>(*this);
}

} // namespace zipper

#endif
