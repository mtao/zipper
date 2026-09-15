#pragma once

#include "zipper/expression/binary/Operation.hpp"
#include "zipper/expression/unary/CoefficientWiseOperation.hpp"
#include "zipper/expression/unary/ScalarOperation.hpp"
#include "zipper/expression/unary/detail/PointwiseOperation.hpp"
#include "zipper/storage/layout_types.hpp"

// Do not include the dense leaves: they include AssignHelper, which consumes
// this classifier. Operation templates above use their actual declarations,
// including any callback constraints, rather than duplicated forward declarations.
namespace zipper::expression::nullary {
template <typename T, typename E, typename L, typename A> class MDArray;
template <typename T, typename E, typename L, typename A> class MDSpan;
template <typename T, index_type... N> class Constant;
template <typename T, int Value, index_type... N> class StaticConstant;
} // namespace zipper::expression::nullary

namespace zipper::expression::detail {

/// A sufficient, never necessary, proof for direct coefficient assignment.
/// Call on raw expressions BEFORE any resize. A true result requires preserving
/// the destination's shape, mapping and storage through evaluation. Same-shape
/// MDArray::resize is harmless; other resizing is not covered by this proof.
/// Only arithmetic leaves/captures are automatic: opaque element copies and
/// conversions may have hidden dependencies. As with ordinary assignment, live
/// valid storage and regular value semantics (no concurrent mutation) are assumed.
/// is_coefficient_consistent is deliberately not an alias-safety guarantee.
template <class From, class To>
auto assignment_is_safe(const From &from, const To &to) -> bool;

namespace assignment_safety {

template <typename T> struct DenseLeaf : std::false_type {};
template <typename T, typename E, typename L, typename A>
struct DenseLeaf<nullary::MDArray<T, E, L, A>>
    : std::bool_constant<
          std::is_arithmetic_v<T> && !std::is_volatile_v<T>
          && std::is_same_v<A, default_accessor_policy<T>>
          && (std::is_same_v<L, storage::layout_left>
              || std::is_same_v<L, storage::layout_right>
#if defined(__cpp_lib_mdspan)
              || std::is_same_v<L, std::layout_stride>
#else
              || std::is_same_v<L, MDSPAN_IMPL_STANDARD_NAMESPACE::layout_stride>
#endif
              )> {
    static constexpr bool writable = !std::is_const_v<T>;
};
template <typename T, typename E, typename L, typename A>
struct DenseLeaf<nullary::MDSpan<T, E, L, A>>
    : DenseLeaf<nullary::MDArray<T, E, L, A>> {};

template <typename From, typename To>
auto same_shape(const From &from, const To &to) -> bool;
template <typename Leaf>
auto valid_storage(const Leaf &leaf) -> bool;

// Closed dispatch: new mapped nodes need an explicit specialization proving
// their effective origin/extents/strides. Merely having expression() or data()
// must never grant permission to traverse an arbitrary node or callback.
template <typename From> struct Source {
    template <typename To>
    static auto check(const From &, const To &) -> bool { return false; }
};

template <typename From>
    requires DenseLeaf<From>::value
struct Source<From> {
    template <typename To>
    static auto check(const From &from, const To &to) -> bool;
};

template <typename T, index_type... N>
struct Source<nullary::Constant<T, N...>> {
    template <typename To>
    static auto check(const nullary::Constant<T, N...> &from, const To &to)
        -> bool;
};
template <typename T, int Value, index_type... N>
struct Source<nullary::StaticConstant<T, Value, N...>> {
    template <typename To>
    static auto check(const nullary::StaticConstant<T, Value, N...> &from,
                      const To &to) -> bool;
};

template <typename Child, typename Op>
struct Source<unary::CoefficientWiseOperation<Child, Op>> {
    template <typename To>
    static auto check(const unary::CoefficientWiseOperation<Child, Op> &from,
                      const To &to) -> bool;
};
template <typename Child, typename Op, typename Scalar, bool Right>
struct Source<unary::ScalarOperation<Child, Op, Scalar, Right>> {
    template <typename To>
    static auto check(const unary::ScalarOperation<Child, Op, Scalar, Right> &from,
                      const To &to) -> bool;
};
template <typename Left, typename Right, typename Op>
struct Source<binary::Operation<Left, Right, Op>> {
    template <typename To>
    static auto check(const binary::Operation<Left, Right, Op> &from,
                      const To &to) -> bool;
};

} // namespace assignment_safety
} // namespace zipper::expression::detail

#include "AssignmentSafety.hxx"
