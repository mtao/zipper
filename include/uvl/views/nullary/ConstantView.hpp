
#if !defined(UVL_VIEWS_NULLARY_CONSTANTVIEW_HPP)
#define UVL_VIEWS_NULLARY_CONSTANTVIEW_HPP

#include "uvl/views/DimensionedViewBase.hpp"

namespace uvl::views {
namespace nullary {
template <typename T, index_type... Indices>
class ConstantView;

}
template <typename T, index_type... Indices>
struct detail::ViewTraits<nullary::ConstantView<T, Indices...>>
: public detail::DefaultViewTraits<T,extents<Indices...>>
{
    constexpr static bool is_coefficient_consistent = true;
};

namespace nullary {
template <typename T, index_type... Indices>
class ConstantView : public DimensionedViewBase<ConstantView<T, Indices...>> {
   public:
    using self_type = ConstantView<T, Indices...>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;
    using value_type = traits::value_type;

    template <typename... Args>
    ConstantView(const value_type& v, Args&&... args)
        : m_value(v), m_extents(std::forward<Args>(args)...) {}
    using Base = DimensionedViewBase<self_type>;
    using Base::extent;

    constexpr const extents_type& extents() const { return m_extents; }

    template <typename... Args>
    value_type coeff(Args&&...) const {
        return m_value;
    }

   private:
    value_type m_value;
    extents_type m_extents;
};  // namespace nullarytemplate<typenameA,typenameB>class AdditionView

template <typename T, index_type... Indices>
ConstantView(const T&, const extents<Indices...>&)
    -> ConstantView<T, Indices...>;

}  // namespace nullary
}  // namespace uvl::views
#endif
