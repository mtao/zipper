
#if !defined(ZIPPER_CONCEPTS_TUPLETYPE_HPP)
#define ZIPPER_CONCEPTS_TUPLETYPE_HPP
#include <tuple>
#pragma GCC diagnostic push
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wdeprecated-literal-operator"
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <range/v3/utility/common_tuple.hpp>
#pragma GCC diagnostic pop


namespace zipper::concepts {
namespace detail {
template <typename>
struct tuple_like : public std::false_type {};

template <typename... Types>
struct tuple_like<std::tuple<Types...>> : public std::true_type {};
template <typename... Types>
struct tuple_like<ranges::common_tuple<Types...>> : public std::true_type {};

}  // namespace detail

template <typename T>
concept TupleLike = detail::tuple_like<T>::value;
}  // namespace zipper::concepts
#endif
