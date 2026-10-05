#pragma once

#include "zipper/static_scalar.hpp"

#include "AssignmentSafety.hpp"

namespace zipper::expression::detail::assignment_safety {

template <typename From, typename To>
auto same_shape(const From &from, const To &to) -> bool {
    if constexpr (From::extents_type::rank() != To::extents_type::rank()) {
        return false;
    } else {
        for (rank_type d = 0; d < From::extents_type::rank(); ++d) {
            if (from.extent(d) != to.extent(d)) {
                return false;
            }
        }
        return true;
    }
}

template <typename Leaf>
auto valid_storage(const Leaf &leaf) -> bool {
    const auto &mapping = leaf.mapping();
    if (!mapping.is_unique() || !mapping.is_strided()) {
        return false;
    }
    for (rank_type d = 0; d < Leaf::extents_type::rank(); ++d) {
        if (leaf.extent(d) == 0) {
            return mapping.required_span_size() == 0;
        }
    }
    const auto &storage = leaf.linear_accessor();
    if (storage.size() == 0 || storage.data() == nullptr) {
        return false;
    }
    // Validate the bounding span against actual backing storage without overflow,
    // before forming any end pointer. Bounds include holes in strided mappings.
    auto remaining = storage.size() - 1;
    if constexpr (Leaf::extents_type::rank() > 0) {
        for (rank_type d = 0; d < Leaf::extents_type::rank(); ++d) {
            const auto stride = mapping.stride(d);
            const auto steps = leaf.extent(d) - 1;
            if (stride == 0 || steps > remaining / stride) {
                return false;
            }
            remaining -= steps * stride;
        }
    }
    return mapping.required_span_size() == storage.size() - remaining;
}

template <typename From>
    requires DenseLeaf<From>::value
template <typename To>
auto Source<From>::check(const From &from, const To &to) -> bool {
    if constexpr (!std::is_same_v<typename From::value_type,
                                  typename To::value_type>
                  || From::extents_type::rank() != To::extents_type::rank()) {
        // No byte-level alias inference across different element types.
        return false;
    } else {
        if (!same_shape(from, to) || !valid_storage(from)) {
            return false;
        }
        const auto count = from.mapping().required_span_size();
        const auto to_count = to.mapping().required_span_size();
        if (count == 0 || to_count == 0) {
            return true;
        }
        const auto *begin = from.linear_accessor().data();
        const auto *to_begin = to.linear_accessor().data();
        const std::less<const void *> less;
        // std::less supplies a total pointer order, including unrelated arrays.
        // Compare complete intervals, never infer independence from begin !=.
        if (!less(to_begin, begin + count) || !less(begin, to_begin + to_count)) {
            return true;
        }
        if (begin != to_begin) {
            return false;
        }
        if constexpr (From::extents_type::rank() > 0) {
            for (rank_type d = 0; d < From::extents_type::rank(); ++d) {
                if (from.mapping().stride(d) != to.mapping().stride(d)) {
                    return false;
                }
            }
        }
        return true;
    }
}

template <typename T, index_type... N>
template <typename To>
auto Source<nullary::Constant<T, N...>>::check(
    const nullary::Constant<T, N...> &from, const To &to) -> bool {
    return std::is_arithmetic_v<T> && !std::is_volatile_v<T>
        && (sizeof...(N) == 0 || same_shape(from, to));
}

template <typename T, T Value, index_type... N>
template <typename To>
auto Source<nullary::StaticConstant<T, Value, N...>>::check(
    const nullary::StaticConstant<T, Value, N...> &from, const To &to) -> bool {
    return std::is_arithmetic_v<T> && !std::is_volatile_v<T>
        && (sizeof...(N) == 0 || same_shape(from, to));
}

template <typename T, index_type... N>
template <typename To>
auto Source<nullary::Zero<T, N...>>::check(const nullary::Zero<T, N...> &from,
                                           const To &to) -> bool {
    return sizeof...(N) == 0 || same_shape(from, to);
}

template <typename Child, typename Op>
template <typename To>
auto Source<unary::CoefficientWiseOperation<Child, Op>>::check(
    const unary::CoefficientWiseOperation<Child, Op> &from, const To &to) -> bool {
    if constexpr (is_pointwise_operation<
                      Op, typename std::remove_cvref_t<Child>::value_type>::value
                  && std::is_arithmetic_v<typename std::remove_cvref_t<
                      decltype(from)>::value_type>) {
        return same_shape(from, to) && assignment_is_safe(from.expression(), to);
    } else {
        return false;
    }
}

template <typename Child, typename Op, typename Scalar, bool Right>
template <typename To>
auto Source<unary::ScalarOperation<Child, Op, Scalar, Right>>::check(
    const unary::ScalarOperation<Child, Op, Scalar, Right> &from,
    const To &to) -> bool {
    using Value = typename std::remove_cvref_t<Child>::value_type;
    using LeftValue = std::conditional_t<Right, Value, Scalar>;
    using RightValue = std::conditional_t<Right, Scalar, Value>;
    if constexpr (std::is_arithmetic_v<Scalar> && !std::is_volatile_v<Scalar>
                  && is_pointwise_operation<Op, LeftValue, RightValue>::value
                  && std::is_arithmetic_v<typename std::remove_cvref_t<
                      decltype(from)>::value_type>) {
        return same_shape(from, to) && assignment_is_safe(from.expression(), to);
    } else {
        return false;
    }
}

template <typename Left, typename Right, typename Op>
template <typename To>
auto Source<binary::Operation<Left, Right, Op>>::check(
    const binary::Operation<Left, Right, Op> &from, const To &to) -> bool {
    if constexpr (is_pointwise_operation<
                      Op, typename std::remove_cvref_t<Left>::value_type,
                      typename std::remove_cvref_t<Right>::value_type>::value
                  && std::is_arithmetic_v<typename std::remove_cvref_t<
                      decltype(from)>::value_type>) {
        return same_shape(from, to) && assignment_is_safe(from.lhs(), to)
            && assignment_is_safe(from.rhs(), to);
    } else {
        return false;
    }
}

} // namespace zipper::expression::detail::assignment_safety

namespace zipper::expression::detail {

template <class From, class To>
auto assignment_is_safe(const From &from, const To &to) -> bool {
    using Destination = std::remove_cvref_t<To>;
    if constexpr (assignment_safety::DenseLeaf<Destination>::value) {
        if constexpr (!assignment_safety::DenseLeaf<Destination>::writable) {
            return false;
        } else {
            return assignment_safety::valid_storage(to)
                && assignment_safety::Source<std::remove_cvref_t<From>>::check(
                    from, to);
        }
    } else {
        return false;
    }
}

} // namespace zipper::expression::detail
