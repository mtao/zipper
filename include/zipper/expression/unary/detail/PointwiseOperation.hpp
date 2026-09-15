#pragma once

#include <functional>
#include <type_traits>

namespace zipper::expression::detail {

/// Semantic opt-in: for these operand types, the operation is pure and
/// index-independent, so reindex(op(child)) == op(reindex(child)). Copying the
/// functor must preserve its behavior. This does not permit reassociation,
/// imply op(0) == 0, or grant alias-free assignment / computed buffer access.
/// Arbitrary callbacks and overloaded arithmetic remain conservative unless
/// explicitly specialized by their author.
template <typename Op, typename... Operands>
struct is_pointwise_operation : std::false_type {};

template <typename T, typename Value>
struct is_pointwise_operation<std::negate<T>, Value>
    : std::bool_constant<
          (std::is_void_v<T> || std::is_arithmetic_v<T>)
          && std::is_arithmetic_v<std::remove_cvref_t<Value>>> {};

template <typename T, typename Left, typename Right>
struct is_pointwise_operation<std::plus<T>, Left, Right>
    : std::bool_constant<
          (std::is_void_v<T> || std::is_arithmetic_v<T>)
          && std::is_arithmetic_v<std::remove_cvref_t<Left>>
          && std::is_arithmetic_v<std::remove_cvref_t<Right>>> {};

template <typename T, typename Left, typename Right>
struct is_pointwise_operation<std::minus<T>, Left, Right>
    : is_pointwise_operation<std::plus<T>, Left, Right> {};

template <typename T, typename Left, typename Right>
struct is_pointwise_operation<std::multiplies<T>, Left, Right>
    : is_pointwise_operation<std::plus<T>, Left, Right> {};

template <typename T, typename Left, typename Right>
struct is_pointwise_operation<std::divides<T>, Left, Right>
    : is_pointwise_operation<std::plus<T>, Left, Right> {};

} // namespace zipper::expression::detail
