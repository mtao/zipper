#if !defined(ZIPPER_CONCEPTS_INDEXARGUMENT_HPP)
#define ZIPPER_CONCEPTS_INDEXARGUMENT_HPP
#include <type_traits>

#include "Index.hpp"
#include "IndexSlice.hpp"

namespace zipper::concepts {

/// A valid argument for an index pack
template <typename T>
concept IndexArgument = IndexSlice<T> || Index<T>;
template <typename... T>
concept IndexArgumentPack = (IndexArgument<T> && ...);

namespace detail {
template <typename T>
struct is_index_argument_pack_tuple : public std::false_type {};

template <IndexArgument... Ts>
struct is_index_argument_pack_tuple<std::tuple<Ts...>> : public std::true_type {
};
} // namespace detail
template <typename T>
concept IndexArgumentPackTuple = detail::is_index_argument_pack_tuple<T>::value;

} // namespace zipper::concepts
#endif
