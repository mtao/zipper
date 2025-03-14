#if !defined(UVL_VIEWS_DETAIL_VIEW_TRAITS_HPP)
#define UVL_VIEWS_DETAIL_VIEW_TRAITS_HPP
#include "uvl/types.hpp"

namespace uvl::views::detail {
template <typename T>
struct ViewTraits;


// NOTE: template parameters should NOT be used in this struct so that derived can overwrite them
template <typename ValueType = void, typename Extents = uvl::dextents<0>>
    struct DefaultViewTraits {
        using value_type = ValueType;
        using extents_type = Extents;

        constexpr static bool is_writable = false;
        // guarantees that V(j) = f(...) cannot depend on V(k) for j != k)
        constexpr static bool is_coefficient_consistent = false;
    };
//{
//    using value_type = ...;
//    using extents_type = ...;
//    using layout_policy = default_layout_policy;
//    using accessor_policy = default_accessor_policy<T>;
//};

}  // namespace uvl::views::detail
#endif
