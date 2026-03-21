#if !defined(ZIPPER_CONCEPTS_DATAARRAY_HPP)
#define ZIPPER_CONCEPTS_DATAARRAY_HPP
#include "Expression.hpp"

namespace zipper::concepts {
namespace detail {

template <typename> struct IsDataArray : std::false_type {};

} // namespace detail

template <typename T>
concept DataArray = detail::IsDataArray<std::remove_cvref_t<T>>::value;

} // namespace zipper::concepts
#endif
