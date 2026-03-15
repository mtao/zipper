#if !defined(ZIPPER_EXPRESSION_CONCEPTS_CAPABILITIES_HPP)
#define ZIPPER_EXPRESSION_CONCEPTS_CAPABILITIES_HPP

#include <concepts>
#include <type_traits>

#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"

namespace zipper::expression::concepts {

/// Checks whether an expression type's `ExpressionTraits` has a particular
/// boolean trait set to true.  All capability concepts below are built on
/// this helper.
///
/// The template parameter `T` may be cv-ref qualified; we always strip
/// qualifiers before looking up traits, which is consistent with how
/// `ExpressionTraits` forwards through const/ref wrappers.

namespace detail {
/// Helper alias to obtain ExpressionTraits for a possibly-qualified type.
/// Stripping cv-ref first avoids relying on the forwarding partial
/// specializations (which do the same thing but one level of indirection
/// deeper).
template <typename T>
using traits_of = ::zipper::expression::detail::ExpressionTraits<
    std::remove_cvref_t<T>>;
} // namespace detail

// ---------------------------------------------------------------------------
// WritableExpression
// ---------------------------------------------------------------------------

/// An expression is **writable** when its coefficients can be mutated through
/// the expression interface.  This is the conjunction of "not const" and
/// "returns references" — i.e. `is_assignable()` at the expression level.
///
/// WritableExpression is the primary concept used in wrapper layers
/// (ZipperBase, VectorBase, etc.) to gate assignment operators, compound
/// assignment, and mutable `operator()`.
///
/// Note: wrapper types (ZipperBase) add an additional `&& !is_const` check
/// for their own template-parameter const-ness.  This concept only checks
/// the **expression's** intrinsic writability.
template <typename T>
concept WritableExpression =
    zipper::concepts::QualifiedExpression<T> &&
    detail::traits_of<T>::is_writable;

// ---------------------------------------------------------------------------
// AssignableExpression
// ---------------------------------------------------------------------------

/// An expression is **assignable** when individual coefficients can be
/// written through `coeff_ref()`.  This is the expression-level predicate
/// `is_assignable() == (!is_const_valued() && is_referrable())`.
///
/// `AssignableExpression` is equivalent to `WritableExpression` for most
/// expression types (since `is_writable` is defined as `is_assignable()`),
/// but the concept is provided separately because certain expression classes
/// (e.g. Slice, Swizzle, Diagonal) use `is_assignable()` directly in their
/// own `coeff_ref` requires clauses.
template <typename T>
concept AssignableExpression =
    zipper::concepts::QualifiedExpression<T> &&
    detail::traits_of<T>::is_assignable();

// ---------------------------------------------------------------------------
// ReferrableExpression
// ---------------------------------------------------------------------------

/// An expression is **referrable** when coefficients can be accessed by
/// reference (const or mutable).  This gates `coeff_ref()`,
/// `const_coeff_ref()`, and determines whether `operator()` returns by
/// reference or by value.
///
/// An expression may be referrable but not assignable (e.g. a const view
/// into an MDArray — `is_reference=true, is_const=true`).
template <typename T>
concept ReferrableExpression =
    zipper::concepts::QualifiedExpression<T> &&
    detail::traits_of<T>::is_referrable();

// ---------------------------------------------------------------------------
// ConstValuedExpression
// ---------------------------------------------------------------------------

/// An expression is **const-valued** when its coefficients are logically
/// immutable.  This is equivalent to `access_features.is_const`.
///
/// Const-valued expressions may still be referrable (returning `const T&`),
/// but they are never assignable.
template <typename T>
concept ConstValuedExpression =
    zipper::concepts::QualifiedExpression<T> &&
    detail::traits_of<T>::is_const_valued();

// ---------------------------------------------------------------------------
// OwningExpression
// ---------------------------------------------------------------------------

/// An expression is **owning** when it (and all sub-expressions) store their
/// data by value rather than by reference.  Owning expressions can safely
/// outlive the scope in which they were created — they are "returnable".
///
/// Non-owning expressions (`stores_references == true`) inherit
/// `NonReturnable` at the wrapper level (ZipperBase), which deletes copy
/// constructors to prevent them from escaping scope as lvalues.  Prvalue
/// returns still work via C++17 guaranteed copy elision.
template <typename T>
concept OwningExpression =
    zipper::concepts::QualifiedExpression<T> &&
    !detail::traits_of<T>::stores_references;

} // namespace zipper::expression::concepts
#endif
