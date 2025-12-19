#if !defined(ZIPPER_CONCEPTS_TUPLETYPE_HPP)
#define ZIPPER_CONCEPTS_TUPLETYPE_HPP
#include <tuple>


namespace zipper::concepts {
namespace detail {
template <typename>
struct tuple_like : public std::false_type {};

template <typename... Types>
struct tuple_like<std::tuple<Types...>> : public std::true_type {};

}  // namespace detail

template <typename T>
concept TupleLike = detail::tuple_like<std::decay_t<T>>::value;

template <typename... T>
concept SingleTuplePackLike = sizeof...(T) == 1 && (TupleLike<T>&&...);
}  // namespace zipper::concepts
#endif
