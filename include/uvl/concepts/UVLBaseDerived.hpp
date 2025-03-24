#if !defined(UVL_CONCEPTS_UVLBASE_DERIVED_HPP)
#define UVL_CONCEPTS_UVLBASE_DERIVED_HPP
#include <type_traits>

#include "ViewDerived.hpp"

namespace uvl {
template <template <concepts::ViewDerived> typename DerivedT,
          concepts::ViewDerived View>
class UVLBase;
}  // namespace uvl
namespace uvl::concepts {
namespace detail {
template <typename>
struct IsUVL : std::false_type {};

template <typename>
struct IsUVLBase : std::false_type {};
template <template <concepts::ViewDerived> typename DerivedT,
          concepts::ViewDerived View>
struct IsUVLBase<UVLBase<DerivedT, View>> : std::true_type {};
}  // namespace detail

template <typename T>
concept UVLBaseDerived = detail::IsUVLBase<T>::value;
}  // namespace uvl::concepts
#endif
