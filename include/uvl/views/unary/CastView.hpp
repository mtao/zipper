
#if !defined(UVL_VIEWS_UNARY_CASTVIEW_HPP)
#define UVL_VIEWS_UNARY_CASTVIEW_HPP

#include "detail/CoeffWiseTraits.hpp"
#include "uvl/views/ViewBase.hpp"
#include "uvl/concepts/ViewDerived.hpp"

namespace uvl::views {
namespace unary {
template <typename A, concepts::ViewDerived B>
class CastView;

}
template <typename A, concepts::ViewDerived B>
struct detail::ViewTraits<unary::CastView<A, B>>
//: public unary::detail::CoeffWiseTraits<A, B> {
//: public detail::ViewTraits<A> {
{
    using Base = detail::ViewTraits<B>;
    using extents_type = typename Base::extents_type;
    using value_type = A;
    using mapping_type = typename Base::mapping_type;
};

namespace unary {
template <typename A, concepts::ViewDerived B>
class CastView : public ViewBase<CastView<A, B>> {
   public:
    using self_type = CastView<A, B>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using mapping_type = traits::mapping_type;
    using value_type = traits::value_type;

    CastView(const B& b) : m_rhs(b) {}
    using Base = ViewBase<self_type>;
    using Base::extent;

    constexpr const extents_type& extents() const { return m_rhs.extents(); }

    template <typename... Args>
    value_type operator()(Args&&... idxs) const {
        return A(m_rhs(std::forward<Args>(idxs)...));
    }

   private:
    const B& m_rhs;
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

}  // namespace unary
}  // namespace uvl::views
#endif
