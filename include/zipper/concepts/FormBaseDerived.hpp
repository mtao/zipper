#if !defined(ZIPPER_CONCEPTS_FORMBASE_DERIVED_HPP)
#define ZIPPER_CONCEPTS_FORMBASE_DERIVED_HPP
#include <concepts>
#include <type_traits>

#include "ExtentsType.hpp"
#include "ViewDerived.hpp"
#include "zipper/types.hpp"

namespace zipper {
template <concepts::QualifiedViewDerived T>
class FormBase;
template <typename ValueType, concepts::ExtentsType Extents,
          bool LeftMajor = true>
class Form_;
template <typename ValueType, concepts::ExtentsType Extents,
          bool LeftMajor = true>
class FormSpan_;
}  // namespace zipper
namespace zipper::concepts {
namespace detail {
template <typename>
struct IsForm : std::false_type {};
template <typename ValueType, concepts::ExtentsType Extents, bool LeftMajor>
struct IsForm<Form_<ValueType, Extents, LeftMajor>> : std::true_type {};

template <typename ValueType, concepts::ExtentsType Extents, bool LeftMajor>
struct IsForm<FormSpan_<ValueType, Extents, LeftMajor>> : std::true_type {};

template <typename>
struct IsFormBase : std::false_type {};
template <typename T>
struct IsFormBase<FormBase<T>> : std::true_type {};
}  // namespace detail

template <typename T>
concept FormBaseDerived =
    (concepts::QualifiedViewDerived<T> && std::derived_from<T, zipper::FormBase<T>>) ||
    detail::IsForm<T>::value || detail::IsFormBase<T>::value;
}  // namespace zipper::concepts
#endif
