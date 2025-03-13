
#if !defined(UVL_VIEWS_UNARY_CASTVIEW_HPP)
#define UVL_VIEWS_UNARY_CASTVIEW_HPP

#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/views/ViewBase.hpp"

namespace uvl::views {
namespace unary {
template <typename A, concepts::ViewDerived B>
class CastView;

template <typename A, concepts::ViewDerived B>
auto cast(const B& b) {
    return CastView<A, B>(b);
}

}  // namespace unary

template <typename A, concepts::ViewDerived B>
struct detail::ViewTraits<unary::CastView<A, B>>
//: public unary::detail::CoeffWiseTraits<A, B> {
//: public detail::ViewTraits<A> {
{
    using Base = detail::ViewTraits<B>;
    using extents_type = typename Base::extents_type;
    using value_type = A;
    constexpr static bool is_writable = false;
};

namespace unary {
template <typename A, concepts::ViewDerived B>
class CastView : public ViewBase<CastView<A, B>> {
   public:
    using self_type = CastView<A, B>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;

    CastView(const CastView&) = default;
    CastView(CastView&&) = default;
    CastView& operator=(const CastView&) = default;
    CastView& operator=(CastView&&) = default;
    CastView(const B& b) : m_rhs(b) {}
    using Base = ViewBase<self_type>;
    using Base::extent;

    constexpr const extents_type& extents() const { return m_rhs.extents(); }

    template <typename... Args>
    value_type coeff(Args&&... idxs) const {
        const auto& value = m_rhs(std::forward<Args>(idxs)...);
        return static_cast<A>(value);
    }

   private:
    const B& m_rhs;
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

}  // namespace unary
}  // namespace uvl::views
#endif
