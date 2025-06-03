#if !defined(ZIPPER_VIEWS_BINARY_FORMTENSORPRODUCTVIEW_HPP)
#define ZIPPER_VIEWS_BINARY_FORMTENSORPRODUCTVIEW_HPP

#include "BinaryViewBase.hpp"
#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/views/binary/TensorProductView.hpp"
#include "zipper/views/unary/PartialTraceView.hpp"
namespace zipper::views {
namespace binary {
template <concepts::ViewDerived A, concepts::ViewDerived B>
class FormTensorProductView;

}
namespace detail {

template <typename A, typename B, typename Seq>
struct form_tensor_partial_trace_type_;

template <typename A, typename B, rank_type... N>
struct form_tensor_partial_trace_type_<A, B,
                                       std::integer_sequence<rank_type, N...>> {
    constexpr static rank_type Off =
        A::extents_type::rank() - B::extents_type::rank();
    using tensor_product_type =
        unary::PartialTraceView<binary::TensorProductView<A, B>>;
    using type = unary::PartialTraceView<tensor_product_type, (N + Off)...>;
};

template <typename A, typename B>
struct form_tensor_partial_trace_type {
    using type = form_tensor_partial_trace_type_<
        A, B,
        decltype(std::make_integer_sequence<
                 rank_type, 2 * B::extents_type::rank()>{})>::type;
};

template <typename A, typename B>
struct ViewTraits<binary::FormTensorProductView<A, B>>
    : public ViewTraits<typename form_tensor_partial_trace_type<A, B>::type> {
    using tensor_product_type = binary::TensorProductView<A, B>;
    using partial_trace_type = form_tensor_partial_trace_type<A, B>::type;
    using tensor_product_traits = ViewTraits<tensor_product_type>;
    using partial_trace_traits = ViewTraits<partial_trace_type>;
    using extents_type = partial_trace_traits::extents_type;
    static_assert(extents_type::rank() ==
                  A::extents_type::rank() - B::extents_type::rank());

    static_assert(extents_type::rank() ==
                  partial_trace_type::extents_type::rank());
    using value_type = partial_trace_traits::value_type;
    using lhs_value_type = tensor_product_traits::lhs_value_type;
    using rhs_value_type = tensor_product_traits::rhs_value_type;
    constexpr static bool holds_extents = false;
    template <typename Derived>
    using base_type = ViewBase<Derived>;
};
}  // namespace detail

namespace binary {
template <concepts::ViewDerived A, concepts::ViewDerived B>
class FormTensorProductView : public ViewBase<FormTensorProductView<A, B>> {
   public:
    using self_type = FormTensorProductView<A, B>;
    using ViewBase<self_type>::operator();
    using traits = zipper::views::detail::ViewTraits<self_type>;
    using value_type = traits::value_type;
    // using Base = BinaryViewBase<self_type, A, B>;
    // using Base::extent;
    // using Base::lhs;
    // using Base::rhs;
    // constexpr static rank_type lhs_rank = traits::lhs_rank;
    // constexpr static rank_type rhs_rank = traits::rhs_rank;

    using extents_type = traits::extents_type;
    static_assert(extents_type::rank() ==
                  A::extents_type::rank() - B::extents_type::rank());
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    //FormTensorProductView(const FormTensorProductView& o)
    //    : m_tensor(o.m_tensor), m_trace(m_tensor) {}
        //: FormTensorProductView(o.m_tensor.lhs(), o.m_tensor.rhs()) {}

    //FormTensorProductView(FormTensorProductView&& o)
    //    : FormTensorProductView(o.m_tensor.lhs(), o.m_tensor.rhs()) {}

    FormTensorProductView() = delete;
     FormTensorProductView(FormTensorProductView&& o) = default;
     FormTensorProductView(const FormTensorProductView& o) = default;
    FormTensorProductView& operator=(FormTensorProductView& o) = delete;
    FormTensorProductView& operator=(FormTensorProductView&& o) = delete;
    FormTensorProductView(const A& a, const B& b)
        : m_tensor(a, b), m_trace(m_tensor) {}

    template <typename... Args>
    value_type coeff(Args&&... args) const {
        typename traits::partial_trace_type trace(m_tensor);
         return trace(std::forward<Args>(args)...);
        //return m_trace(std::forward<Args>(args)...);
    }
    const extents_type& extents() const { return m_trace.extents(); }

   private:
    traits::tensor_product_type m_tensor;
    traits::partial_trace_type m_trace;
};

template <concepts::ViewDerived A, concepts::ViewDerived B>
FormTensorProductView(const A& a, const B& b) -> FormTensorProductView<A, B>;
}  // namespace binary
}  // namespace zipper::views
#endif
