#if !defined(ZIPPER_CONCEPTS_CONTAINER_HPP)
#define ZIPPER_CONCEPTS_CONTAINER_HPP
#include "Expression.hpp"

namespace zipper::concepts {
namespace detail {

template <typename> struct IsContainer : std::false_type {};

} // namespace detail

template <typename T>
concept Container = detail::IsContainer<std::remove_cvref_t<T>>::value;

} // namespace zipper::concepts
#endif
