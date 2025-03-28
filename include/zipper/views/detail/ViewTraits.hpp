#if !defined(ZIPPER_VIEWS_DETAIL_VIEW_TRAITS_HPP)
#define ZIPPER_VIEWS_DETAIL_VIEW_TRAITS_HPP
#include "zipper/types.hpp"

namespace zipper::views::detail {
template <typename T>
struct ViewTraits;

// NOTE: template parameters should NOT be used in this struct so that derived
// can overwrite them
template <typename ValueType = void, typename Extents = zipper::dextents<0>>
struct DefaultViewTraits {
    using value_type = ValueType;
    using extents_type = Extents;

    constexpr static bool is_writable = false;
    // guarantees that V(j) = f(...) cannot depend on V(k) for j != k)
    constexpr static bool is_coefficient_consistent = false;

    // only depends on values passed in
    constexpr static bool is_value_based = false;
};
//{
//    using value_type = ...;
//    using extents_type = ...;
//    using layout_policy = default_layout_policy;
//    using accessor_policy = default_accessor_policy<T>;
//};

}  // namespace zipper::views::detail
#endif
