
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
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = typename traits::extents_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;

    AdditionView(const A& a, const B& b)
        requires(extents_traits::is_static)
        : m_lhs(a), m_rhs(b) {}
    AdditionView(const A& a, const B& b)
        requires(!extents_traits::is_static)
        : m_lhs(a), m_rhs(b), m_extents(a.extent(0)) {}
    using Base = DimensionedViewBase<self_type>;
    using Base::extent;
    const extents_type& extents() const { return m_extents; }

    using ViewBase<self_type>::operator();
    template <typename... Args>
    auto coeff(Args&&... idxs) const
        requires((std::is_convertible_v<Args, index_type> && ...))
    {
        static_assert(sizeof...(idxs) == extents_type::rank());
        static_assert(A::extents_type::rank() == extents_type::rank());
        static_assert(B::extents_type::rank() == extents_type::rank());
        return m_lhs(idxs...) + m_rhs(idxs...);
    }

   private:
    const A& m_lhs;
    const B& m_rhs;
    extents_type m_extents;
};  // namespace binarytemplate<typenameA,typenameB>class AdditionView

template <concepts::ViewDerived A, concepts::ViewDerived B>
AdditionView(const A& a, const B& b) -> AdditionView<A, B>;
}  // namespace binary
}  // namespace uvl::views
#endif
