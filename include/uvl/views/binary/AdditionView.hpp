
#if !defined(UVL_VIEWS_BINARY_ADDITIONVIEW_HPP)
#define UVL_VIEWS_BINARY_ADDITIONVIEW_HPP

#include "detail/CoeffWiseTraits.hpp"

namespace uvl::views {
namespace binary {
template <ViewDerived A, ViewDerived B>
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
    using mapping_type = typename Base::mapping_type;
};

namespace binary {
template <ViewDerived A, ViewDerived B>
class AdditionView  // : public StaticViewBase<AdditionView<A, B>>
{
   public:
    using self_type = AdditionView<A, B>;
    AdditionView(const A& a, const B& b) : m_lhs(a), m_rhs(b) {}
    using Base = StaticViewBase<self_type>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    // using Base = MappedViewBase<self_type>;
    // using Base::extent;
    // using Base::extents;

    // using value_type = traits::value_type;
    //  using extents_type = traits::extents_type;
    //  using extents_traits = uvl::detail::ExtentsTraits<extents_type>;

    // const mapping_type& mapping() const { return derived().mapping(); }
    // const extents_type& extents() const { return derived().extents(); }
    // template <typename... Args>
    // auto operator()(Args&&... idxs) const {
    //    return m_lhs(get_index(std::forward<Args>(idxs)...)) +
    //           m_rhs(get_index(std::forward<Args>(idxs)...));
    //}

   private:
    const A& m_lhs;
    const B& m_rhs;
};  // namespace binarytemplate<typenameA,typenameB>class AdditionView

template <ViewDerived A, ViewDerived B>
AdditionView(const A& a, const B& b) -> AdditionView<A, B>;
}  // namespace binary
}  // namespace uvl::views
#endif
