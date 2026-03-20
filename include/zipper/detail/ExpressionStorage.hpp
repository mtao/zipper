#if !defined(ZIPPER_DETAIL_EXPRESSIONSTORAGE_HPP)
#define ZIPPER_DETAIL_EXPRESSIONSTORAGE_HPP

#include "ViewPropagating.hpp"

#include <type_traits>

namespace zipper::detail {

/// Determines how to store an expression child based on value category.
///
/// When T is an lvalue reference (T&), the referent outlives us and we store
/// a reference — unless the decayed type is view-propagating, in which case
/// we store by value (copying is cheap, just copies a reference wrapper,
/// and the caller has asserted lifetime safety via ref()).
///
/// When T is a non-reference type (deduced from an rvalue), we
/// store by value to prevent dangling.
///
/// Usage with forwarding references:
///   template <typename U>
///   Foo(U&& expr) : m_expr(std::forward<U>(expr)) {}
///   expression_storage_t<U> m_expr;
///
/// When called as Foo(lvalue), U = T& → storage = T& (reference).
///   Exception: if T is view-propagating, storage = T (by value).
/// When called as Foo(rvalue), U = T  → storage = T  (by value).
template <typename T>
using expression_storage_t = std::conditional_t<
    std::is_lvalue_reference_v<T> && !ViewPropagating<T>,
    T,                          // lvalue ref, not view-propagating: store as reference
    std::remove_reference_t<T>  // rvalue or view-propagating: store by value (own it)
>;

/// The const-qualified version of the stored type, suitable for read-only
/// access to the stored expression.
template <typename T>
using const_expression_storage_t = std::conditional_t<
    std::is_lvalue_reference_v<T> && !ViewPropagating<T>,
    const std::remove_reference_t<T> &,
    const std::remove_reference_t<T>
>;

} // namespace zipper::detail

#endif
