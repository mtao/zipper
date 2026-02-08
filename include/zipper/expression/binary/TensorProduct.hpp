#if !defined(ZIPPER_EXPRESSION_BINARY_TENSORPRODUCT_HPP)
#define ZIPPER_EXPRESSION_BINARY_TENSORPRODUCT_HPP

#include "BinaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/pack_index.hpp"

namespace zipper::expression {
namespace binary {
template <zipper::concepts::QualifiedExpression A,
          zipper::concepts::QualifiedExpression B>
class TensorProduct;

namespace _detail_tensor {

template <typename A, typename B>
struct tensor_coeffwise_extents_values;
template <index_type... A, index_type... B>
struct tensor_coeffwise_extents_values<extents<A...>, extents<B...>> {
    using product_extents_type = zipper::extents<A..., B...>;
    using a_extents_type = extents<A...>;
    using b_extents_type = extents<B...>;

    static_assert(product_extents_type::rank() ==
                  a_extents_type::rank() + b_extents_type::rank());

    using a_extents_traits = zipper::detail::ExtentsTraits<a_extents_type>;
    using b_extents_traits = zipper::detail::ExtentsTraits<b_extents_type>;

    template <rank_type... N, rank_type... M>
    constexpr static product_extents_type merge(
        const extents<A...>& a, const extents<B...>& b,
        std::integer_sequence<rank_type, N...>,
        std::integer_sequence<rank_type, M...>) {
        using AEI = a_extents_traits::dynamic_indices_helper;
        using BEI = b_extents_traits::dynamic_indices_helper;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
        constexpr static auto ADI = AEI::dynamic_indices;
        constexpr static auto BDI = BEI::dynamic_indices;
#pragma GCC diagnostic pop


        return product_extents_type(a.extent(ADI[N])..., b.extent(BDI[M])...);
    }

    constexpr static product_extents_type merge(const extents<A...>& a,
                                                const extents<B...>& b) {
        return merge(
            a, b,
            std::make_integer_sequence<rank_type, a_extents_type::rank_dynamic()>{},
            std::make_integer_sequence<rank_type, b_extents_type::rank_dynamic()>{}

            );
    }
};

}  // namespace _detail_tensor
}  // namespace binary

template <concepts::QualifiedExpression A, concepts::QualifiedExpression B>
struct detail::ExpressionTraits<binary::TensorProduct<A, B>>
    : public binary::detail::DefaultBinaryExpressionTraits<A, B>
{
    using ATraits = detail::ExpressionTraits<A>;
    constexpr static rank_type lhs_rank = ATraits::extents_type::rank();
    using BTraits = detail::ExpressionTraits<B>;
    constexpr static rank_type rhs_rank = BTraits::extents_type::rank();
    using CEV = binary::_detail_tensor::tensor_coeffwise_extents_values<
        typename ATraits::extents_type, typename BTraits::extents_type>;

    static_assert(std::is_same_v<typename CEV::a_extents_type,
                                 typename ATraits::extents_type>);
    static_assert(std::is_same_v<typename CEV::b_extents_type,
                                 typename BTraits::extents_type>);
    using extents_type = CEV::product_extents_type;
    static_assert(
        std::is_same_v<typename CEV::product_extents_type, extents_type>);
    static_assert(extents_type::rank() == lhs_rank + rhs_rank);

    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
};

namespace binary {
template <zipper::concepts::QualifiedExpression A,
          zipper::concepts::QualifiedExpression B>
class TensorProduct : public BinaryExpressionBase<TensorProduct<A, B>, A, B> {
   public:
    using self_type = TensorProduct<A, B>;
    using Base = BinaryExpressionBase<self_type, A, B>;
    using ExpressionBase = expression::ExpressionBase<self_type>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using value_type = traits::value_type;
    using Base::extent;
    using Base::lhs;
    using Base::rhs;
    constexpr static rank_type lhs_rank = traits::lhs_rank;
    constexpr static rank_type rhs_rank = traits::rhs_rank;

    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    TensorProduct(const TensorProduct&) = default;
    TensorProduct(TensorProduct&&) = default;
    TensorProduct& operator=(const TensorProduct&) = delete;
    TensorProduct& operator=(TensorProduct&&) = delete;
    TensorProduct(const A& a, const B& b)
        requires(extents_traits::is_static)
        : Base(a, b) {}
    TensorProduct(const A& a, const B& b)
        requires(!extents_traits::is_static)
        : Base(a, b, traits::CEV::merge(a.extents(), b.extents())) {}

    template <typename... Args, rank_type... ranks>
    auto lhs_value(std::integer_sequence<rank_type, ranks...>,
                   Args&&... args) const -> decltype(auto) {
        return lhs()(
            zipper::detail::pack_index<ranks>(std::forward<Args>(args)...)...);
    }
    template <typename... Args, rank_type... ranks>
    auto rhs_value(std::integer_sequence<rank_type, ranks...>,
                   Args&&... args) const -> decltype(auto) {
        return rhs()(zipper::detail::pack_index<ranks + lhs_rank>(
            std::forward<Args>(args)...)...);
    }

    template <typename... Args>
    value_type coeff(Args&&... args) const {
        return lhs_value(std::make_integer_sequence<rank_type, lhs_rank>{},
                         std::forward<Args>(args)...) *
               rhs_value(std::make_integer_sequence<rank_type, rhs_rank>{},
                         std::forward<Args>(args)...);
    }
};

template <zipper::concepts::Expression A, zipper::concepts::Expression B>
TensorProduct(const A& a, const B& b) -> TensorProduct<A, B>;

}  // namespace binary
}  // namespace zipper::expression
#endif
