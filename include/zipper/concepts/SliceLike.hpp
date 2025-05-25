#if !defined(ZIPPER_CONCEPTS_SLICELIKE_HPP)
#define ZIPPER_CONCEPTS_SLICELIKE_HPP
#include "IndexLike.hpp"
#include "zipper/types.hpp"

namespace zipper::concepts {

namespace detail {
template <typename T>
struct is_slice : public std::false_type {};

template <IndexLike OffsetType, IndexLike ExtentType, IndexLike StrideType>
struct is_slice<slice_type<OffsetType, ExtentType, StrideType>>
    : public std::true_type {};

template <typename T>
struct slice_like : public index_like<T> {};

template <IndexLike OffsetType, IndexLike ExtentType, IndexLike StrideType>
struct slice_like<slice_type<OffsetType, ExtentType, StrideType>>
    : public std::true_type {};

template <>
struct slice_like<full_extent_type> : public std::true_type {};

}  // namespace detail

// A value ois slice-like if it can belong in a slice.
// That is, it's an index, or a component in a tuple
template <typename T>
concept SliceLike = detail::slice_like<std::decay_t<T>>::value;
}  // namespace zipper::concepts
#endif
