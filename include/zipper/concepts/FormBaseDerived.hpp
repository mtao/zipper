#if !defined(ZIPPER_CONCEPTS_FORMBASE_DERIVED_HPP)
#define ZIPPER_CONCEPTS_FORMBASE_DERIVED_HPP
#include <concepts>
#include <type_traits>

#include "ViewDerived.hpp"
#include "zipper/types.hpp"

namespace zipper {
template <concepts::ViewDerived T>
class FormBase;
template <typename T, index_type... Indices>
class Form;
}  // namespace zipper
namespace zipper::concepts {
namespace detail {
template <typename>
struct IsForm : std::false_type {};
template <typename T, index_type... Indices>
struct IsForm<Form<T, Indices...>> : std::true_type {};

template <typename>
struct IsFormBase : std::false_type {};
template <typename T>
struct IsFormBase<FormBase<T>> : std::true_type {};
}  // namespace detail

template <typename T>
concept FormBaseDerived =
    (concepts::ViewDerived<T> && std::derived_from<T, zipper::FormBase<T>>) ||
    detail::IsForm<T>::value || detail::IsFormBase<T>::value;
}  // namespace zipper::concepts
#endif
