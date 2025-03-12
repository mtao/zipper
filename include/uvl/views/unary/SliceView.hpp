


#if !defined(UVL_VIEWS_UNARY_RESHAPEVIEW_HPP)
#define UVL_VIEWS_UNARY_RESHAPEVIEW_HPP

#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/views/MappedViewBase.hpp"

namespace uvl::views {
namespace unary {

template <concepts::ViewDerived B, typename SliceType>
class SliceView;

}
template <concepts::ViewDerived B, typename SliceType>
struct detail::ViewTraits<
    unary::SliceView<B, SliceType>>
{
    using Base = detail::ViewTraits<B>;
    using BaseExtentsTraits =
        uvl::detail::ExtentsTraits<typename Base::extents_type>;

    using mapping_type = decltype(std::experimental::submdspan_mapping(B,SliceType{}));
    //using extents_type = Extents;
    //using value_type = Base::value_type;
    //using mapping_type = LayoutPolicy::template mapping<extents_type>;
    //constexpr static bool is_writable = Base::is_writable;
};

namespace unary {
template <concepts::ViewDerived B, typename SliceType>
class SliceView : public MappedViewBase<
                        SliceView<B, SliceType>> {
   public:
    using self_type = SliceView<B, SliceType>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using mapping_type = traits::mapping_type;
    using value_type = traits::value_type;

    using Base = MappedViewBase<self_type>;
    using Base::extent;
    //SliceView(const B& b, const Extents& e = {}) : Base(e), m_view(b) {}

    template <typename... Args>
    value_type coeff(Args&&... idxs) const {
        //index_type index = get_index(idxs...);
        //if constexpr (concepts::MappedViewDerived<B>) {
        //    const auto& value = m_view.coeff(index);
        //} else if constexpr (concepts::VectorViewDerived<B>) {
        //    return m_view(index);
        //}
    }
    template <typename... Args>
    value_type& coeff_ref(Args&&... idxs) {
        //index_type index = get_index(idxs...);
        //if constexpr (concepts::MappedViewDerived<B>) {
        //    const auto& value = m_view.coeff_ref(index);
        //} else if constexpr (concepts::VectorViewDerived<B>) {
        //    return m_view(index);
        //}
    }

   private:
    const B& m_view;
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

}  // namespace unary
}  // namespace uvl::views
#endif
