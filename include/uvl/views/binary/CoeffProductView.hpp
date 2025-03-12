

#if !defined(UVL_VIEWS_BINARY_COEFFPRODUCTVIEW_HPP)
#define UVL_VIEWS_BINARY_COEFFPRODUCTVIEW_HPP

#include "detail/CoeffWiseTraits.hpp"
#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/views/MappedViewBase.hpp"

namespace uvl::views {
namespace binary {
template <concepts::ViewDerived A, concepts::ViewDerived B>
class CoeffProductView;

}
template <typename A, typename B>
struct detail::ViewTraits<binary::CoeffProductView<A, B>>
//: public binary::detail::CoeffWiseTraits<A, B> {
//: public detail::ViewTraits<A> {
{
    using Base = detail::ViewTraits<A>;
    using extents_type = typename Base::extents_type;
    using value_type = typename Base::value_type;
    using mapping_type = typename Base::mapping_type;
    constexpr static bool is_writable = false;
};

namespace binary {
template <concepts::ViewDerived A, concepts::ViewDerived B>
class CoeffProductView : public MappedViewBase<CoeffProductView<A, B>> {
   public:
    using self_type = CoeffProductView<A, B>;
    CoeffProductView(const A& a, const B& b) : m_lhs(a), m_rhs(b) {}
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using Base = MappedViewBase<self_type>;
    using Base::extent;
    using Base::extents;

    // using value_type = traits::value_type;
    //  using extents_type = traits::extents_type;
    //  using extents_traits = uvl::detail::ExtentsTraits<extents_type>;

    // const mapping_type& mapping() const { return derived().mapping(); }
    // const extents_type& extents() const { return derived().extents(); }
    template <typename... Args>
    auto coeff(Args&&... idxs) const {
        return m_lhs(idxs...) * m_rhs(idxs...);
    }

   private:
    const A& m_lhs;
    const B& m_rhs;
};  // namespace binarytemplate<typenameA,typenameB>class CoeffProductView

template <concepts::ViewDerived A, concepts::ViewDerived B>
CoeffProductView(const A& a, const B& b) -> CoeffProductView<A, B>;
}  // namespace binary
}  // namespace uvl::views
#endif
