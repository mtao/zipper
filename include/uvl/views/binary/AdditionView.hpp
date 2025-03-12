
#if !defined(UVL_VIEWS_BINARY_ADDITIONVIEW_HPP)
#define UVL_VIEWS_BINARY_ADDITIONVIEW_HPP

#include "detail/CoeffWiseTraits.hpp"
#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/views/DimensionedViewBase.hpp"

namespace uvl::views {
namespace binary {
template <concepts::ViewDerived A, concepts::ViewDerived B>
class AdditionView;

}
template <typename A, typename B>
struct detail::ViewTraits<binary::AdditionView<A, B>>
//: public binary::detail::CoeffWiseTraits<A, B> {
//: public detail::ViewTraits<A> {
{
    using Base = detail::ViewTraits<A>;
    using extents_type = typename Base::extents_type;
    using value_type = typename Base::value_type;
    constexpr static bool is_writable = false;
};

namespace binary {
template <concepts::ViewDerived A, concepts::ViewDerived B>
class AdditionView : public DimensionedViewBase<AdditionView<A, B>> {
   public:
    using self_type = AdditionView<A, B>;
    AdditionView(const A& a, const B& b) : m_lhs(a), m_rhs(b) {}
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using Base = DimensionedViewBase<self_type>;
    using Base::extent;
    using Base::extents;

    using ViewBase<self_type>::operator();
    template <typename... Args>
    auto coeff(Args&&... idxs) const
        requires((std::is_convertible_v<Args, index_type> && ...))
    {
        return m_lhs(idxs...) + m_rhs(idxs...);
    }

   private:
    const A& m_lhs;
    const B& m_rhs;
};  // namespace binarytemplate<typenameA,typenameB>class AdditionView

template <concepts::ViewDerived A, concepts::ViewDerived B>
AdditionView(const A& a, const B& b) -> AdditionView<A, B>;
}  // namespace binary
}  // namespace uvl::views
#endif
