#if !defined(ZIPPER_CONCEPTS_INDEX_HPP)
#define ZIPPER_CONCEPTS_INDEX_HPP
#include <tuple>
#include <type_traits>

namespace zipper::concepts {
namespace detail {
template <typename T> struct is_index : public std::is_integral<T> {};

template <typename T, T v>
struct is_index<std::integral_constant<T, v>> : public std::true_type {};

} // namespace detail

/// A single index value / coordinate to pass into an MDArray
/// An index is either an integer or integer constant
template <typename T>
concept Index = detail::is_index<std::decay_t<T>>::value;

/// A variadic set of arguments that all specify a single index (not a slice)
template <typename... T>
concept IndexPack = (Index<T> && ...);

namespace detail {
template <typename T> struct is_index_pack_tuple : public std::false_type {};

template <Index... Ts>
struct is_index_pack_tuple<std::tuple<Ts...>> : public std::true_type {};
} // namespace detail
//
/// A tuple of indices for which every value is single index (not a slice)
/// useful for determining if an IndexArgument specifies a single value or not
template <typename T>
concept IndexPackTuple = detail::is_index_pack_tuple<T>::value;
} // namespace zipper::concepts
#endif
