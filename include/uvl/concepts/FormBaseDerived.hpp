#if !defined(UVL_CONCEPTS_FORMBASE_DERIVED_HPP)
#define UVL_CONCEPTS_FORMBASE_DERIVED_HPP
#include <concepts>
#include <type_traits>

#include "ViewDerived.hpp"
#include "uvl/types.hpp"

namespace uvl {
template <concepts::ViewDerived T>
class FormBase;
template <typename T, index_type... Indices>
class Form;
}  // namespace uvl
namespace uvl::concepts {
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
    (concepts::ViewDerived<T> && std::derived_from<T, uvl::FormBase<T>>) ||
    detail::IsForm<T>::value || detail::IsFormBase<T>::value;
}  // namespace uvl::concepts
#endif
