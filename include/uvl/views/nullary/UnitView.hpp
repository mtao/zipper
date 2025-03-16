#if !defined(UVL_VIEWS_NULLARY_UNITVIEW_HPP)
#define UVL_VIEWS_NULLARY_UNITVIEW_HPP

#include <utility>

#include "NullaryViewBase.hpp"
#include "uvl/concepts/TupleLike.hpp"
#include "uvl/views/DimensionedViewBase.hpp"

namespace uvl::views {
namespace nullary {
template <typename T, index_type Extent, typename IndexType>
class UnitView;

}
template <typename T, index_type Extent, typename IndexType>
struct detail::ViewTraits<nullary::UnitView<T, Extent, IndexType>>
    : public nullary::detail::DefaultNullaryViewTraits<T, Extent> {
    constexpr static bool is_value_based = false;
};

namespace nullary {
template <typename T, index_type Extent, typename IndexType>
class UnitView : public DimensionedViewBase<UnitView<T, Extent, IndexType>> {
   public:
    using self_type = UnitView<T, Extent, IndexType>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;
    using value_type = traits::value_type;
    constexpr static bool dynamic_index = std::is_same_v<IndexType, index_type>;
    constexpr static bool dynamic_size = extents_traits::is_dynamic;

    static_assert(
        dynamic_index ||
        !dynamic_size);  // if we have a dynamic size we disallow indices

    UnitView(const UnitView&) = default;
    UnitView(UnitView&&) = default;
    UnitView& operator=(const UnitView&) = default;
    UnitView& operator=(UnitView&&) = default;

    UnitView()
        requires(extents_traits::is_static && dynamic_index)
    = default;

    UnitView(const index_type size)
        : m_extents(extent)
              requires(dynamic_size && !dynamic_index)
    {}
    UnitView(const index_type extent, IndexType index)
        : m_extents(extent), m_index(index) {}
    using Base = DimensionedViewBase<self_type>;
    using Base::extent;

    constexpr const extents_type& extents() const { return m_extents; }

    template <typename... Args>
    value_type coeff(Args&&... idxs) const {
        if constexpr (sizeof...(Args) == 1 &&
                      (concepts::TupleLike<Args> && ...)) {
            return (std::get<0>(idxs) == m_index && ...);
        } else {
            return ((idxs == m_index) && ...);
        }
    }

   private:
    extents_type m_extents;
    IndexType m_index;
};  // namespace nullarytemplate<typenameA,typenameB>class AdditionView

template <typename T, index_type size, index_type index>
auto unit_vector() {
    return UnitView<T, size, std::integral_constant<index_type, index>>{};
}
template <typename T>
auto unit_vector(index_type size, index_type index) {
    return UnitView<T, std::dynamic_extent, index_type>{size, index};
}

template <typename T, index_type size>
auto unit_vector(index_type index) requires(size != std::dynamic_extent>{
    return UnitView<T, size, index_type>{index};
}

}  // namespace nullary
}  // namespace uvl::views
#endif
