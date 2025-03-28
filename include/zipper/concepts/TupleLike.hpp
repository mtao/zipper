
#if !defined(ZIPPER_CONCEPTS_TUPLETYPE_HPP)
#define ZIPPER_CONCEPTS_TUPLETYPE_HPP
#include <tuple>
#include <range/v3/utility/common_tuple.hpp>

namespace zipper::concepts {
namespace detail {
template <typename>
struct tuple_like : public std::false_type {};

template <typename... Types>
struct tuple_like<std::tuple<Types...>> : public std::true_type {};
template <typename... Types>
struct tuple_like<ranges::common_tuple<Types...>> : public std::true_type {};

template <typename A, typename B>
struct tuple_like<std::pair<A, B>> : public std::true_type {};

template <typename A, std::size_t N>
struct tuple_like<std::array<A, N>> : public std::true_type {};
}  // namespace detail

template <typename T>
concept TupleLike = detail::tuple_like<T>::value;
}  // namespace zipper::concepts
#endif
