#pragma once

#include <cmath>
#include <concepts>
#include <limits>
#include <type_traits>

#include <zipper/concepts/Algebraic.hpp>
#include <zipper/types.hpp>

namespace zipper::utils::scalar_math {

namespace detail {

using std::abs;
using std::pow;
using std::sqrt;

template <typename T>
auto adl_abs(const T &value) -> decltype(abs(value)) {
    return abs(value);
}

template <typename T>
auto adl_sqrt(const T &value) -> decltype(sqrt(value)) {
    return sqrt(value);
}

template <typename T>
auto adl_pow(const T &value, const T &exponent)
    -> decltype(pow(value, exponent)) {
    return pow(value, exponent);
}

} // namespace detail

template <typename T>
auto absolute_value(const T &value) -> T {
    return T(detail::adl_abs(value));
}

template <typename T>
auto square_root(const T &value) -> T {
    return T(detail::adl_sqrt(value));
}

template <typename T>
auto root(const T &value, index_type degree) -> T {
    return T(detail::adl_pow(value, T(1) / T(degree)));
}

template <typename T>
auto copy_sign(const T &magnitude, const T &sign) -> T {
    const T positive = absolute_value(magnitude);
    return sign < T{0} ? -positive : positive;
}

template <typename T>
auto hypotenuse(const T &lhs, const T &rhs) -> T {
    const T a = absolute_value(lhs);
    const T b = absolute_value(rhs);
    const T scale = a < b ? b : a;
    if (scale == T{0}) { return T{0}; }
    const T x = a / scale;
    const T y = b / scale;
    return scale * square_root(x * x + y * y);
}

template <typename T>
struct traits {
    static auto epsilon() -> T
        requires std::numeric_limits<T>::is_specialized
    {
        return std::numeric_limits<T>::epsilon();
    }
};

template <typename T>
auto epsilon() -> T
    requires requires { { traits<T>::epsilon() } -> std::same_as<T>; }
{
    return traits<T>::epsilon();
}

template <typename T>
concept orthogonal_decomposition_scalar =
    concepts::field<T> && std::totally_ordered<T> &&
    requires(const T &value) {
        { T{0} } -> std::same_as<T>;
        { T{1} } -> std::same_as<T>;
        { T{2} } -> std::same_as<T>;
        { detail::adl_abs(value) } -> std::convertible_to<T>;
        { detail::adl_sqrt(value) } -> std::convertible_to<T>;
        { epsilon<T>() } -> std::same_as<T>;
    };

} // namespace zipper::utils::scalar_math
