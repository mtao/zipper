#if !defined(ZIPPER_VIEWS_NULLARY_UNITVIEW_HPP)
#define ZIPPER_VIEWS_NULLARY_UNITVIEW_HPP

#include <utility>

#include "NullaryViewBase.hpp"
#include "zipper/concepts/TupleLike.hpp"
#include "zipper/views/DimensionedViewBase.hpp"

namespace zipper::views {
namespace nullary {
template <typename T, index_type Extent, typename IndexType>
class UnitView;

}
template <typename T, index_type Extent, typename IndexType>
struct detail::ViewTraits<nullary::UnitView<T, Extent, IndexType>>
    : public nullary::detail::DefaultNullaryViewTraits<T, Extent> {
    constexpr static bool is_value_based = false;

    // returns true if a dimension is sparse, false if dense
    consteval static bool is_sparse(rank_type) { return true; }
};

namespace nullary {
template <typename T, index_type Extent, typename IndexType>
class UnitView
    : public NullaryViewBase<UnitView<T, Extent, IndexType>, T, Extent> {
   public:
    using self_type = UnitView<T, Extent, IndexType>;
    using traits = zipper::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    using value_type = traits::value_type;
    using Base = NullaryViewBase<UnitView<T, Extent, IndexType>, T, Extent>;
    constexpr static bool dynamic_index = std::is_same_v<IndexType, index_type>;
    constexpr static bool dynamic_size = extents_traits::is_dynamic;

    using Base::extent;
    static_assert(
        dynamic_index ||
        !dynamic_size);  // if we have a dynamic size we disallow indices

    UnitView(const UnitView&) = default;
    UnitView(UnitView&&) = default;
    UnitView& operator=(const UnitView&) = default;
    UnitView& operator=(UnitView&&) = default;

    UnitView()
        requires(extents_traits::is_static && !dynamic_index)
    = default;

    UnitView(const index_type index)
        requires(!dynamic_size && dynamic_index)
        : m_index(index) {}
    UnitView(const index_type my_extent, IndexType index)
        : Base(extents_type(my_extent)), m_index(index) {}

    template <typename T_>
    value_type coeff(const T_& idx) const {
        if (idx == m_index) {
            return 1;
        } else {
            return 0;
        }
    }

    template <rank_type R>
        requires(R == 0)
    constexpr std::array<index_type, 1> nonZeros(index_type) const {
        return std::array<index_type, 1>{{m_index}};
    }

   private:
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
auto unit_vector(index_type index)
    requires(size != std::dynamic_extent)
{
    return UnitView<T, size, index_type>(index);
}

}  // namespace nullary
}  // namespace zipper::views
#endif
