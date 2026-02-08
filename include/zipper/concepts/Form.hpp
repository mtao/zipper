#if !defined(ZIPPER_CONCEPTS_FORM_HPP)
#define ZIPPER_CONCEPTS_FORM_HPP
#include "Expression.hpp"

namespace zipper {} // namespace zipper
namespace zipper::concepts {
namespace detail {
template <typename> struct IsForm : std::false_type {};

} // namespace detail

template <typename T>
concept Form =
    detail::IsForm<std::remove_cvref_t<T>>::value;

} // namespace zipper::concepts
#endif
