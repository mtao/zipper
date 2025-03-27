#if !defined(UVL_CONCEPTS_SLICELIKE_HPP)
#define UVL_CONCEPTS_SLICELIKE_HPP
#include "IndexLike.hpp"
#include "IndexList.hpp"
#include "uvl/types.hpp"
#include <vector>
#include <array>
#include <map>

namespace uvl::concepts {

namespace detail {
template <typename T>
struct slice_like : public index_like<T> {};

template <IndexLike OffsetType, IndexLike ExtentType, IndexLike StrideType>
struct slice_like<slice_type<OffsetType, ExtentType, StrideType>>
    : public std::true_type {};

template <>
struct slice_like<full_extent_type> : public std::true_type {};

template <IndexList L>
struct slice_like<L> : public std::true_type {};

template <typename T>
struct slice_like<std::initializer_list<T>> : public std::true_type {};

}  // namespace detail

// A value ois slice-like if it can belong in a slice.
// That is, it's an index, or a component in a tuple
template <typename T>
concept SliceLike = detail::slice_like<T>::value;
}  // namespace uvl::concepts
#endif
