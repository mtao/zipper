#if !defined(UVL_CONCEPTS_INDEXLIKE_HPP)
#define UVL_CONCEPTS_INDEXLIKE_HPP
#include <type_traits>

namespace uvl::concepts {
namespace detail {
template <typename T>
struct index_like : public std::is_integral<T> {};

template <typename T, T index>
struct index_like<std::integral_constant<T, index>> : public std::true_type {};

}  // namespace detail

template <typename T>
concept IndexLike = detail::index_like<T>::value;
}  // namespace uvl::concepts
#endif
