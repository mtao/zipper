#if !defined(ZIPPER_CONCEPTS_SLICE_HPP)
#define ZIPPER_CONCEPTS_SLICE_HPP
#include "Expression.hpp"
#include "Index.hpp"
#include "zipper/types.hpp"
#include <array>
#include <vector>

namespace zipper::concepts {

namespace detail {

template <typename T> struct is_index_slice : public std::false_type {};

template <Index OffsetType, Index ExtentType, Index StrideType>
struct is_index_slice<slice_type<OffsetType, ExtentType, StrideType>>
    : public std::true_type {};

template <> struct is_index_slice<full_extent_t> : public std::true_type {};

template <>
struct is_index_slice<std::vector<index_type>> : public std::true_type {};

template <size_t N>
struct is_index_slice<std::array<index_type, N>> : public std::true_type {};

/// expressions that store an index-friendly type and a single dimension are
/// slices
template <zipper::concepts::Expression Expression>
  requires(Expression::extents_type::rank() == 1 &&
           Index<typename Expression::value_type>)
struct is_index_slice<Expression> : public std::true_type {};

} // namespace detail

/// An argument for indices that can store more than one value in a single
/// dimension of a MDArray
template <typename T>
concept IndexSlice = detail::is_index_slice<std::decay_t<T>>::value;

} // namespace zipper::concepts
#endif
