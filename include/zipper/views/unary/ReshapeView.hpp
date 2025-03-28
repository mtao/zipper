

#if !defined(ZIPPER_VIEWS_UNARY_RESHAPEVIEW_HPP)
#define ZIPPER_VIEWS_UNARY_RESHAPEVIEW_HPP

#include "zipper/concepts/LinearView.hpp"
#include "zipper/views/MappedViewBase.hpp"

namespace zipper::views {
namespace unary {

template <concepts::LinearView B, typename Extents,
          typename LayoutPolicy = zipper::default_layout_policy,
          typename AccessorPolicy =
              zipper::default_accessor_policy<typename B::value_type>>
class ReshapeView;

}
template <concepts::LinearView B, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
struct detail::ViewTraits<
    unary::ReshapeView<B, Extents, LayoutPolicy, AccessorPolicy>>
//: public unary::detail::CoeffWiseTraits<A, B> {
//: public detail::ViewTraits<A> {
{
    using Base = detail::ViewTraits<B>;
    using BaseExtentsTraits =
        zipper::detail::ExtentsTraits<typename Base::extents_type>;
    using extents_type = Extents;
    using value_type = Base::value_type;
    using mapping_type = LayoutPolicy::template mapping<extents_type>;
    constexpr static bool is_writable = Base::is_writable;
};

namespace unary {
template <concepts::LinearView B, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
class ReshapeView : public MappedViewBase<
                        ReshapeView<B, Extents, LayoutPolicy, AccessorPolicy>> {
   public:
    using self_type = ReshapeView<B, Extents, LayoutPolicy, AccessorPolicy>;
    using traits = zipper::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using mapping_type = traits::mapping_type;
    using value_type = traits::value_type;

    using Base = MappedViewBase<self_type>;
    using Base::extent;
    ReshapeView(const B& b, const Extents& e = {}) : Base(e), m_view(b) {}

    template <typename... Args>
    value_type coeff(Args&&... idxs) const {
        index_type index = get_index(idxs...);
        if constexpr (concepts::MappedViewDerived<B>) {
            const auto& value = m_view.coeff(index);
        } else if constexpr (concepts::VectorViewDerived<B>) {
            return m_view(index);
        }
    }
    template <typename... Args>
    value_type& coeff_ref(Args&&... idxs) {
        index_type index = get_index(idxs...);
        if constexpr (concepts::MappedViewDerived<B>) {
            const auto& value = m_view.coeff_ref(index);
        } else if constexpr (concepts::VectorViewDerived<B>) {
            return m_view(index);
        }
    }

   private:
    const B& m_view;
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

}  // namespace unary
}  // namespace zipper::views
#endif
