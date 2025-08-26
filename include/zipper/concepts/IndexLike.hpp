#if !defined(ZIPPER_CONCEPTS_INDEXLIKE_HPP)
#define ZIPPER_CONCEPTS_INDEXLIKE_HPP
#include <type_traits>

namespace zipper::concepts {
namespace detail {
template <typename T>
struct index_like : public std::is_integral<T> {};

template <typename T, T index>
struct index_like<std::integral_constant<T, index>> : public std::true_type {};

}  // namespace detail

template <typename T>
concept IndexLike = detail::index_like<std::decay_t<T>>::value;
}  // namespace zipper::concepts
#endif
