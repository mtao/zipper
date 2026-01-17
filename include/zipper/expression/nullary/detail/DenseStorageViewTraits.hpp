#if !defined(ZIPPER_VIEWS_DETAIL_PLAINOBJECTVIEWTRAITS_HPP)
#define ZIPPER_VIEWS_DETAIL_PLAINOBJECTVIEWTRAITS_HPP
#include "zipper/views/detail/ViewTraits.hpp"
#include "zipper/detail//ExtentsTraits.hpp"
#include "zipper/types.hpp"

namespace zipper::views::nullary {

template <typename T>
class DenseStorageViewBase;
}
namespace zipper::views::detail {
template <typename T>
concept StorageViewDerived = std::derived_from<
    T, nullary::DenseStorageViewBase<
           T>>;
template <typename T>
struct DenseStorageViewTraits;
//{
//    using value_type = ...;
//    using extents_type = ...;
//    using layout_policy = default_layout_policy;
//    using accessor_policy = default_accessor_policy<T>;
//};
// template <StorageViewDerived T>
// struct ViewTraits<T> : public StorageViewTraits<T> {
//    using Base = StorageViewTraits<T>;
//    using extents_type = Base::extents_type;
//    using extents_traits = Base::extents_traits;
//    using layout_policy = Base::layout_policy;
//    using mapping_type = typename layout_policy::template
//    mapping<extents_type>;
//};

template <typename Derived>
using mdspan_type = zipper::mdspan<typename ViewTraits<Derived>::value_type,
                                typename ViewTraits<Derived>::Extents,
                                typename ViewTraits<Derived>::layout_policy,
                                typename ViewTraits<Derived>::accessor_policy>;
}  // namespace zipper::views::detail
#endif
