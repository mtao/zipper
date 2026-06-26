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

// ---------------------------------------------------------------------------
// HasLayoutMapping / LinearArray
// ---------------------------------------------------------------------------

/// An expression **has a layout mapping** when it exposes `mapping()` — a
/// tensor-index → linear-offset map with per-dimension `stride()` / `extents()`.
/// Such an expression can be *linearized* (its access expressed as offsets)
/// rather than walked through the recursive `coeff` chain. This is the pure
/// reindexing capability: it does not imply an addressable buffer (a lazy
/// `2*A` can have a mapping with no storage of its own).
template <typename T>
concept HasLayoutMapping =
    zipper::concepts::QualifiedExpression<T> &&
    detail::traits_of<T>::has_layout_mapping;

/// An expression is a **linear array** when it has a layout mapping AND a flat
/// unchecked `operator[](index_type)` over a contiguous buffer, satisfying
/// `e[ e.mapping()(i,j) ] == e(i,j)`. Dense storage and buffer-sharing views
/// (Slice, Swizzle) are linear arrays; value-computing expressions are not.
///
/// Consumers that read raw storage (e.g. the blocked GEMM pack) gate on this.
template <typename T>
concept LinearArray =
    zipper::concepts::QualifiedExpression<T> &&
    detail::traits_of<T>::is_linear_array;

// ---------------------------------------------------------------------------
// FlatVectorizable
// ---------------------------------------------------------------------------

/// An expression is **flat-vectorizable** when it exposes a flat,
/// value-returning `operator[](index_type)` that enumerates its coefficients in
/// the same (buffer/row-major) order a contiguous `LinearArray` would. Such an
/// expression can be assigned with a single linear loop
/// `for k: to[k] = from[k]`, which the compiler auto-vectorizes — no per-element
/// `mapping()`/stride math (that hides the unit stride and defeats the
/// vectorizer), no recursive `coeff` walk.
///
/// This is the RECURSIVE, structural sibling of `LinearArray`:
///   - a `LinearArray` leaf is flat-vectorizable (its `operator[]` is the buffer);
///   - a coefficient-wise op (`2*A`, `A+B`, cast) is flat-vectorizable **iff its
///     operands are** — its `operator[]` composes the op over the operands'
///     flat accessors, bottoming out at contiguous leaves.
/// So an arbitrarily deep coeff-wise tree (`4*A+B`, `A+B+C+D`, `4*(A+B)`)
/// satisfies this exactly when every leaf is contiguous.
///
/// This is a DECLARED trait (`ExpressionTraits::flat_layout_type`), not a
/// structural `operator[]` probe: an expression is flat-vectorizable iff its
/// whole tree has a CONSISTENT contiguous mapping (every leaf shares one
/// layout). It is layout-GENERIC — row-major or column-major both qualify, as
/// long as the tree agrees. Reordering views (transpose/Swizzle, shape-changing
/// Slice) carry a buffer but enumerate out of layout order, so they leave
/// `flat_layout_type` void and are excluded.
template <typename T>
concept FlatVectorizable =
    zipper::concepts::QualifiedExpression<T> &&
    !std::is_void_v<typename detail::traits_of<T>::flat_layout_type>;

/// Two expressions are **flat-compatible** when both are flat-vectorizable AND
/// share the SAME contiguous layout, so a single linear loop
/// `for k: dst[k] = src[k]` visits the same logical element on both sides
/// (given equal extents). This is the soundness gate for the vectorized
/// linear-assignment fast path — it holds for any layout the two agree on, not
/// just row-major.
template <typename Dst, typename Src>
concept FlatCompatible =
    FlatVectorizable<Dst> && FlatVectorizable<Src> &&
    std::is_same_v<typename detail::traits_of<Dst>::flat_layout_type,
                   typename detail::traits_of<Src>::flat_layout_type>;

} // namespace zipper::expression::concepts
#endif
