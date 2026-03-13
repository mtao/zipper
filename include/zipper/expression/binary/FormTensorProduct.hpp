#if !defined(ZIPPER_EXPRESSION_BINARY_FORMTENSORPRODUCT_HPP)
#define ZIPPER_EXPRESSION_BINARY_FORMTENSORPRODUCT_HPP


#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/assert.hpp"
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
    // Off = max(0, a_rank - b_rank) but since rank_type is unsigned,
    // we need to be careful about underflow
    constexpr static rank_type a_rank = std::decay_t<A>::extents_type::rank();
    constexpr static rank_type b_rank = std::decay_t<B>::extents_type::rank();
    constexpr static rank_type Off = (a_rank >= b_rank) ? (a_rank - b_rank) : 0;
    using tensor_product_type =
        binary::TensorProduct<A, B>;
    // The PartialTrace stores a reference to the TensorProduct member
    // inside FormTensorProduct, so use reference-qualified type.
    using partial_trace_type = unary::PartialTrace<tensor_product_type&, (N + Off)...>;
};

template <typename A, typename B>
struct form_tensor_partial_trace_type {
    constexpr static rank_type a_rank = std::decay_t<A>::extents_type::rank();
    constexpr static rank_type b_rank = std::decay_t<B>::extents_type::rank();
    // Contract min(a_rank, b_rank) pairs of indices
    constexpr static rank_type contracted_rank = (a_rank < b_rank) ? a_rank : b_rank;

    using helper = form_tensor_partial_trace_type_<
        A, B,
        decltype(std::make_integer_sequence<
                 rank_type, 2 * contracted_rank>{})>;
    using partial_trace_type = typename helper::partial_trace_type;
    using tensor_product_type = typename helper::tensor_product_type;

    // When a_rank >= b_rank, result is a form (or scalar)
    // When a_rank < b_rank, result is a tensor
    constexpr static bool result_is_form = (a_rank >= b_rank);
    constexpr static rank_type result_rank = (a_rank >= b_rank) ? (a_rank - b_rank) : (b_rank - a_rank);
};

}  // namespace _detail_form_tensor

template <concepts::QualifiedExpression A, concepts::QualifiedExpression B>
struct detail::ExpressionTraits<binary::FormTensorProduct<A, B>>
    : public detail::ExpressionTraits<typename _detail_form_tensor::form_tensor_partial_trace_type<A, B>::partial_trace_type> {
    using _helper = _detail_form_tensor::form_tensor_partial_trace_type<A, B>;
    using tensor_product_type = typename _helper::tensor_product_type;
    using partial_trace_type = typename _helper::partial_trace_type;
    using tensor_product_traits = detail::ExpressionTraits<tensor_product_type>;
    using partial_trace_traits = detail::ExpressionTraits<partial_trace_type>;
    using extents_type = partial_trace_traits::extents_type;
    static_assert(extents_type::rank() == _helper::result_rank);

    static_assert(extents_type::rank() ==
                  partial_trace_type::extents_type::rank());
    using value_type = partial_trace_traits::value_type;

    constexpr static bool result_is_form = _helper::result_is_form;
};

namespace binary {
template <zipper::concepts::QualifiedExpression A, zipper::concepts::QualifiedExpression B>
class FormTensorProduct : public ExpressionBase<FormTensorProduct<A, B>> {
   public:
    using self_type = FormTensorProduct<A, B>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using value_type = traits::value_type;

    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    constexpr static bool result_is_form = traits::result_is_form;

    FormTensorProduct() = delete;
    FormTensorProduct(const FormTensorProduct& o)
        : FormTensorProduct(o.m_tensor.lhs(), o.m_tensor.rhs()) {}
    FormTensorProduct& operator=(FormTensorProduct& o) = delete;
    FormTensorProduct& operator=(FormTensorProduct&& o) = delete;
    FormTensorProduct(const std::decay_t<A>& a, const std::decay_t<B>& b)
        : m_tensor(a, b), m_trace(m_tensor) {
        ZIPPER_ASSERT(&m_trace.expression() == &m_tensor);
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

    /// Recursively deep-copy children so the result owns all data.
    auto make_owned() const {
        auto owned_a = m_tensor.lhs().make_owned();
        auto owned_b = m_tensor.rhs().make_owned();
        return FormTensorProduct<const decltype(owned_a), const decltype(owned_b)>(
            std::move(owned_a), std::move(owned_b));
    }

   private:
    traits::tensor_product_type m_tensor;
    traits::partial_trace_type m_trace;
};

template <zipper::concepts::Expression A, zipper::concepts::Expression B>
FormTensorProduct(const A& a, const B& b) -> FormTensorProduct<const A&, const B&>;

}  // namespace binary
}  // namespace zipper::expression
#endif
