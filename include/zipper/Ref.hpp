#pragma once

#include "concepts/Zipper.hpp"

namespace zipper {

/// Create a view-propagating reference to a mutable Zipper lvalue.
///
/// The caller asserts that `x` outlives all views derived from the
/// returned wrapper.  The result — and any views derived from it
/// (head, tail, row, transpose, etc.) — are Returnable.
///
/// Usage:
///   auto r = ref(x);            // r is Returnable
///   auto h = ref(x).head<3>();  // h is also Returnable
///
///   template <concepts::Vector A>
///   auto f(const A& a) { return a.head<3>(); }
///   auto h2 = f(ref(x));        // h2 is Returnable
template <concepts::Zipper T>
auto ref(T &x) { return x.ref(); }

/// Rvalue overload is deleted — ref() requires an lvalue with a
/// guaranteed lifetime.
template <concepts::Zipper T>
void ref(const T &&) = delete;

/// Create a view-propagating const reference to a Zipper lvalue.
///
/// Like ref(), but the result is read-only.  Binds to both const and
/// mutable lvalues.
template <concepts::Zipper T>
auto cref(const T &x) { return x.ref(); }

/// Rvalue overload is deleted.
template <concepts::Zipper T>
void cref(const T &&) = delete;

} // namespace zipper
