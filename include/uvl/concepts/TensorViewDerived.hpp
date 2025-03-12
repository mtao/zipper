#if !defined(UVL_CONCEPTS_TENSOR_VIEW_DERIVED_HPP)
#define UVL_CONCEPTS_TENSOR_VIEW_DERIVED_HPP
#include <type_traits>

#include "ViewDerived.hpp"
#include "uvl/types.hpp"
namespace uvl::views::detail {
template <typename T>
class ViewTraits;
}
namespace uvl::concepts {

template <typename T, uvl::rank_type rank>
concept TensorViewDerived =
    ViewDerived<T> &&
    views::detail::ViewTraits<T>::extents_type::rank() == rank;
}  // namespace uvl::concepts
#endif
