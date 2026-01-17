#if !defined(ZIPPER_CONCEPTS_TENSOR_VIEW_DERIVED_HPP)
#define ZIPPER_CONCEPTS_TENSOR_VIEW_DERIVED_HPP
#include <type_traits>

#include "ViewDerived.hpp"
#include "zipper/types.hpp"
namespace zipper::views::detail {
template <typename T> struct ViewTraits;
}
namespace zipper::concepts {

template <typename T, zipper::rank_type rank>
concept TensorView =
    QualifiedView<T> &&
    views::detail::ViewTraits<std::decay_t<T>>::extents_type::rank() == rank;
} // namespace zipper::concepts
#endif
