#if !defined(ZIPPER_CONCEPTS_ARRAY_HPP)
#define ZIPPER_CONCEPTS_ARRAY_HPP
#include "Expression.hpp"

namespace zipper::concepts {
namespace detail {

template <typename> struct IsArray : std::false_type {};

} // namespace detail

template <typename T>
concept Array =
    detail::IsArray<std::remove_cvref_t<T>>::value;

} // namespace zipper::concepts
#endif
