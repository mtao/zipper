#if !defined(UVL_VIEWS_BINARY_FORMTENSORPRODUCTVIEW_HPP)
#define UVL_VIEWS_BINARY_FORMTENSORPRODUCTVIEW_HPP

#include "BinaryViewBase.hpp"
#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/detail/pack_index.hpp"

#include "uvl/views/unary/PartialTraceView.hpp"
#include "uvl/views/binary/TensorProductView.hpp"
namespace uvl::views {
namespace binary {
template <concepts::ViewDerived A, concepts::ViewDerived B>
class FormTensorProductView;

}
namespace detail {


}  // namespace detail
template <typename A, typename B>
struct detail::ViewTraits<binary::FormTensorProductView<A, B>>
    : public detail::ViewTraits<unary::PartialTraceView<binary::FormTensorProductView<A,B>>> {

    constexpr static bool holds_extents = false;
    template <typename Derived>
    using base_type = ViewBase<Derived>;
    };

namespace binary {
template <concepts::ViewDerived A, concepts::ViewDerived B>
class FormTensorProductView : public BinaryViewBase<FormTensorProductView<A, B>, A, B> {
   public:
    using self_type = FormTensorProductView<A, B>;
    using ViewBase<self_type>::operator();
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using value_type = traits::value_type;
    using Base = BinaryViewBase<self_type, A, B>;
    using Base::extent;
    using Base::lhs;
    using Base::rhs;
    constexpr static rank_type lhs_rank = traits::lhs_rank;
    constexpr static rank_type rhs_rank = traits::rhs_rank;

    using extents_type = traits::extents_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;

    FormTensorProductView(const A& a, const B& b): m_tensor(a,b), m_trace(m_tensor) {}

    template <typename... Args>
    value_type coeff(Args&&... args) const {
        return m_trace(std::forward<Args>(args)...);
    }
    const extents_type& extents() const
    {
        return m_trace.extents();
    }

    binary::TensorProductView<A,B> m_tensor;
    unary::PartialTraceView<binary::TensorProductView<A,B>> m_trace;
};  // namespace binarytemplate<typenameA,typenameB>class WedgeProductView

template <concepts::ViewDerived A, concepts::ViewDerived B>
FormTensorProductView(const A& a, const B& b) -> FormTensorProductView<A, B>;
}  // namespace binary
}  // namespace uvl::views
#endif
