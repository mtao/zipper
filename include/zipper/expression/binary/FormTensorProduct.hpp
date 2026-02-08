#if !defined(ZIPPER_EXPRESSION_BINARY_FORMTENSORPRODUCT_HPP)
#define ZIPPER_EXPRESSION_BINARY_FORMTENSORPRODUCT_HPP


#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/expression/ExpressionBase.hpp"
#include "zipper/expression/binary/TensorProduct.hpp"
#include "zipper/expression/unary/PartialTrace.hpp"
namespace zipper::expression {

namespace binary {
template <zipper::concepts::QualifiedExpression A, zipper::concepts::QualifiedExpression B>
class FormTensorProduct;

}

namespace _detail_form_tensor {

template <typename A, typename B, typename Seq>
struct form_tensor_partial_trace_type_;

template <typename A, typename B, rank_type... N>
struct form_tensor_partial_trace_type_<A, B,
                                       std::integer_sequence<rank_type, N...>> {
    constexpr static rank_type Off =
        A::extents_type::rank() - B::extents_type::rank();
    using tensor_product_type =
        binary::TensorProduct<A, B>;
    using partial_trace_type = unary::PartialTrace<tensor_product_type, (N + Off)...>;
};

template <typename A, typename B>
struct form_tensor_partial_trace_type {

    using helper = form_tensor_partial_trace_type_<
        A, B,
        decltype(std::make_integer_sequence<
                 rank_type, 2 * B::extents_type::rank()>{})>;
    using partial_trace_type = typename helper::partial_trace_type;
    using tensor_product_type = typename helper::tensor_product_type;
};

}  // namespace _detail_form_tensor

template <concepts::QualifiedExpression A, concepts::QualifiedExpression B>
struct detail::ExpressionTraits<binary::FormTensorProduct<A, B>>
    : public detail::ExpressionTraits<typename _detail_form_tensor::form_tensor_partial_trace_type<A, B>::partial_trace_type> {
    using tensor_product_type = _detail_form_tensor::form_tensor_partial_trace_type<A, B>::tensor_product_type;
    using partial_trace_type = _detail_form_tensor::form_tensor_partial_trace_type<A, B>::partial_trace_type;
    using tensor_product_traits = detail::ExpressionTraits<tensor_product_type>;
    using partial_trace_traits = detail::ExpressionTraits<partial_trace_type>;
    using extents_type = partial_trace_traits::extents_type;
    static_assert(extents_type::rank() ==
                  A::extents_type::rank() - B::extents_type::rank());

    static_assert(extents_type::rank() ==
                  partial_trace_type::extents_type::rank());
    using value_type = partial_trace_traits::value_type;
};

namespace binary {
template <zipper::concepts::QualifiedExpression A, zipper::concepts::QualifiedExpression B>
class FormTensorProduct : public ExpressionBase<FormTensorProduct<A, B>> {
   public:
    using self_type = FormTensorProduct<A, B>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using value_type = traits::value_type;

    using extents_type = traits::extents_type;
    static_assert(extents_type::rank() ==
                  A::extents_type::rank() - B::extents_type::rank());
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    FormTensorProduct() = delete;
    FormTensorProduct(const FormTensorProduct& o)
        : FormTensorProduct(o.m_tensor.lhs(), o.m_tensor.rhs()) {}
    FormTensorProduct& operator=(FormTensorProduct& o) = delete;
    FormTensorProduct& operator=(FormTensorProduct&& o) = delete;
    FormTensorProduct(const A& a, const B& b)
        : m_tensor(a, b), m_trace(m_tensor) {
        assert(&m_trace.expression() == &m_tensor);
    }

    template <typename... Args>
    value_type coeff(Args&&... args) const {
        return m_trace(std::forward<Args>(args)...);
    }
    constexpr auto extent(rank_type i) const -> index_type {
        return m_trace.extent(i);
    }
    constexpr auto extents() const -> extents_type {
        return extents_traits::make_extents_from(*this);
    }

   private:
    traits::tensor_product_type m_tensor;
    traits::partial_trace_type m_trace;
};

template <zipper::concepts::QualifiedExpression A, zipper::concepts::QualifiedExpression B>
FormTensorProduct(const A& a, const B& b) -> FormTensorProduct<A, B>;

}  // namespace binary
}  // namespace zipper::expression
#endif
