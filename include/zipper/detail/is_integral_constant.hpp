
#if !defined(ZIPPER_DETAIL_IS_INTEGRAL_CONSTANT_HPP)
#define ZIPPER_DETAIL_IS_INTEGRAL_CONSTANT_HPP
#include <type_traits>

namespace zipper::detail {

template <typename T>
struct is_integral_constant: public std::false_type {};


template <typename T, T value>
struct is_integral_constant<std::integral_constant<T,value>>: public std::true_type{};

template <typename T>
constexpr bool is_integral_constant_v = is_integral_constant<T>::value;
}
#endif
