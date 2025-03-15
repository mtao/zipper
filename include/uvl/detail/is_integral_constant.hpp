
#if !defined(UVL_DETAIL_IS_INTEGRAL_CONSTANT_HPP)
#define UVL_DETAIL_IS_INTEGRAL_CONSTANT_HPP
#include <type_traits>

namespace uvl::detail {

template <typename T>
struct is_integral_constant: public std::false_type {};


template <typename T, T value>
struct is_integral_constant<std::integral_constant<T,value>>: public std::true_type{};

template <typename T>
constexpr bool is_integral_constant_v = is_integral_constant<T>::value;
}
#endif
