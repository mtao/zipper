#if !defined(ZIPPER_DETAIL_EXTENTS_CHECK_HPP)
#define ZIPPER_DETAIL_EXTENTS_CHECK_HPP

/// @file extents_check.hpp
/// @brief Always-on validation for runtime extents vs static extents.
///
/// Unlike `ZIPPER_ASSERT` which is debug-only, these checks are always
/// active to catch programming errors like `Vector<double, 3>(5)` in
/// release builds.

#include <stdexcept>
#include <format>

#include "zipper/types.hpp"
#include "ExtentsTraits.hpp"

namespace zipper::detail {

/// Validate that a runtime extent value matches a static extent.
/// No-op when the static extent is `dynamic_extent`.
/// Throws `std::invalid_argument` on mismatch.
inline void check_extent(index_type runtime_value,
                         index_type static_extent_value,
                         rank_type dim) {
  if (static_extent_value != dynamic_extent &&
      runtime_value != static_extent_value) {
    throw std::invalid_argument(std::format(
        "extent mismatch: dimension {} has static extent {} but was "
        "constructed with runtime value {}",
        dim, static_extent_value, runtime_value));
  }
}

/// Validate all runtime extents against static extents of a given
/// extents type.  Only checks dimensions with static (non-dynamic) extents.
///
/// Usage:
///   check_extents<extents_type>(row_val, col_val, ...);
///
/// The number of arguments must match the rank of the extents type.
template <concepts::Extents Extents, typename... Args>
  requires(sizeof...(Args) == Extents::rank())
void check_extents(Args... args) {
  index_type vals[] = {static_cast<index_type>(args)...};
  for (rank_type d = 0; d < Extents::rank(); ++d) {
    check_extent(vals[d], Extents::static_extent(d), d);
  }
}

} // namespace zipper::detail

#endif // ZIPPER_DETAIL_EXTENTS_CHECK_HPP
