#pragma once

#include <type_traits>

#include "zipper/expression/binary/Operation.hpp"
#include "zipper/expression/unary/CoefficientWiseOperation.hpp"
#include "zipper/expression/unary/ScalarOperation.hpp"
#include "zipper/expression/unary/detail/PointwiseOperation.hpp"
#include "zipper/types.hpp"

// Do not include the dense leaves: they include AssignHelper, which consumes
// this classifier. Operation templates above use their actual declarations,
// including any callback constraints, rather than duplicated forward declarations.
namespace zipper::expression::nullary {
template <typename T, typename E, typename L, typename A> class MDArray;
template <typename T, index_type... N> class Constant;
template <typename T, T Value, index_type... N> class StaticConstant;
template <typename T, index_type... N> class Zero;
} // namespace zipper::expression::nullary

/// @file AssignmentSafety.hpp
/// @brief Compile-time proof that `to = from` may be written in place.
///
/// AssignHelper evaluates straight into the destination when this proof
/// holds, and otherwise through a snapshot (a temporary). The proof is made
/// from types alone -- it never inspects addresses -- so it costs nothing at
/// run time. It holds when every coefficient of `from` is computed by
/// pointwise operations (the opted-in, index-independent arithmetic of
/// PointwiseOperation.hpp) from:
///
///   * generators that read no storage (Constant, StaticConstant, Zero):
///     safe into any destination, views included;
///   * owning dense arrays (MDArray), provided the destination is itself an
///     owning array. Two distinct owning arrays never share storage, and
///     when they are the same object a pointwise expression reads only the
///     coefficient it writes (`a = a + b`).
///
/// Everything else -- views (MDSpan, slices, transposes), products,
/// reductions, callbacks -- may read storage the destination writes, so it is
/// not proven; assignment snapshots it. When the caller knows the operands
/// are independent, `dest.noalias() = expr` skips the snapshot. Only
/// arithmetic, non-volatile element types are covered: other element copies
/// may have hidden dependencies.
namespace zipper::expression::detail::assignment_safety {

template <typename T>
inline constexpr bool plain_v =
    std::is_arithmetic_v<T> && !std::is_volatile_v<T>;

/// How a source expression reads storage, as far as the proof can tell.
///   provable:      every coefficient comes from the cases above;
///   reads_storage: some leaf is an owning array (not only generators).
/// Closed dispatch: a node is only traversed through an explicit
/// specialization here; merely exposing expression() or data() must never
/// grant permission to look inside an arbitrary node or callback.
template <typename From>
struct Source {
    static constexpr bool provable = false;
    static constexpr bool reads_storage = true;
};

template <typename From>
using source_t = Source<std::remove_cvref_t<From>>;

/// An owning dense array: its storage cannot be shared with another object.
template <typename T>
struct OwningLeaf : std::false_type {};
template <typename T, typename E, typename L, typename A>
struct OwningLeaf<nullary::MDArray<T, E, L, A>>
    : std::bool_constant<plain_v<T>
                         && std::is_same_v<A, default_accessor_policy<T>>> {};

template <typename From>
    requires OwningLeaf<From>::value
struct Source<From> {
    static constexpr bool provable = true;
    static constexpr bool reads_storage = true;
};

/// Generators read no storage: no write can change what they evaluate to.
template <typename T, index_type... N>
struct Source<nullary::Constant<T, N...>> {
    static constexpr bool provable = plain_v<T>;
    static constexpr bool reads_storage = false;
};
template <typename T, T Value, index_type... N>
struct Source<nullary::StaticConstant<T, Value, N...>> {
    static constexpr bool provable = plain_v<T>;
    static constexpr bool reads_storage = false;
};
template <typename T, index_type... N>
struct Source<nullary::Zero<T, N...>> {
    static constexpr bool provable = true;
    static constexpr bool reads_storage = false;
};

template <typename Child, typename Op>
struct Source<unary::CoefficientWiseOperation<Child, Op>> {
    using value_type =
        typename unary::CoefficientWiseOperation<Child, Op>::value_type;
    static constexpr bool provable =
        is_pointwise_operation<
            Op, typename std::remove_cvref_t<Child>::value_type>::value
        && plain_v<value_type> && source_t<Child>::provable;
    static constexpr bool reads_storage = source_t<Child>::reads_storage;
};

template <typename Child, typename Op, typename Scalar, bool Right>
struct Source<unary::ScalarOperation<Child, Op, Scalar, Right>> {
    using Value = typename std::remove_cvref_t<Child>::value_type;
    using value_type =
        typename unary::ScalarOperation<Child, Op, Scalar, Right>::value_type;
    static constexpr bool provable =
        plain_v<Scalar>
        && is_pointwise_operation<Op,
                                  std::conditional_t<Right, Value, Scalar>,
                                  std::conditional_t<Right, Scalar, Value>>::value
        && plain_v<value_type> && source_t<Child>::provable;
    static constexpr bool reads_storage = source_t<Child>::reads_storage;
};

template <typename Left, typename Right, typename Op>
struct Source<binary::Operation<Left, Right, Op>> {
    using value_type = typename binary::Operation<Left, Right, Op>::value_type;
    static constexpr bool provable =
        is_pointwise_operation<
            Op, typename std::remove_cvref_t<Left>::value_type,
            typename std::remove_cvref_t<Right>::value_type>::value
        && plain_v<value_type> && source_t<Left>::provable
        && source_t<Right>::provable;
    static constexpr bool reads_storage =
        source_t<Left>::reads_storage || source_t<Right>::reads_storage;
};

/// Destination element type, or void when To has none.
template <typename To>
struct destination_value {
    using type = void;
};
template <typename To>
    requires requires { typename To::value_type; }
struct destination_value<To> {
    using type = typename To::value_type;
};

} // namespace zipper::expression::detail::assignment_safety

namespace zipper::expression::detail {

/// `to = from` may be evaluated straight into `to` (see the file docs).
/// The destination may be resized to from's shape first: generators read no
/// storage, and an owning destination either is a distinct object from every
/// owning source or the same object, which has from's shape already.
template <class From, class To>
inline constexpr bool assignment_is_safe_v = [] {
    namespace as = assignment_safety;
    using Destination = std::remove_cvref_t<To>;
    using Value = typename as::destination_value<Destination>::type;
    using S = as::source_t<From>;
    if constexpr (!as::plain_v<Value> || std::is_const_v<Value>
                  || !S::provable) {
        return false;
    } else if constexpr (!S::reads_storage) {
        return true;
    } else {
        return as::OwningLeaf<Destination>::value;
    }
}();

/// Value form of assignment_is_safe_v, for deduced argument types.
template <class From, class To>
constexpr auto assignment_is_safe(const From &, const To &) -> bool {
    return assignment_is_safe_v<From, To>;
}

} // namespace zipper::expression::detail
