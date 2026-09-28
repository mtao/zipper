#pragma once

#include <type_traits>

namespace zipper::detail {

/// Whether `.noalias()` may target an expression of type T.
///
/// Opt-in only: each expression header that has a known dense assignment
/// contract specializes this at the bottom of its own header. In particular,
/// sparse destinations must retain their support-building path and never opt
/// in.
///
/// - Dense leaves (MDArray, MDSpan) opt in when their element type is mutable.
/// - Views whose coefficient writes forward directly to their child (index
///   remapping only) opt in via `forwards_noalias_v<Child>`, so support
///   composes recursively down to a mutable dense leaf.
template <typename T>
inline constexpr bool supports_noalias_v = false;

/// Helper for write-forwarding views: true when the (possibly reference- or
/// const-qualified) child is mutable and itself supports noalias.
template <typename Child>
inline constexpr bool forwards_noalias_v =
    !std::is_const_v<std::remove_reference_t<Child>> &&
    supports_noalias_v<std::remove_cvref_t<Child>>;

} // namespace zipper::detail
