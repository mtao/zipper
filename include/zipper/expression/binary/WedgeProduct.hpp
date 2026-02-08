#if !defined(ZIPPER_EXPRESSION_BINARY_WEDGEPRODUCT_HPP)
#define ZIPPER_EXPRESSION_BINARY_WEDGEPRODUCT_HPP

#include "BinaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/pack_index.hpp"

namespace zipper::expression {
namespace binary {
template <zipper::concepts::Expression A, zipper::concepts::Expression B>
class WedgeProduct;

}
namespace _detail_wedge {

template <typename A, typename B>
struct wedge_coeffwise_extents_values;
template <index_type... A, index_type... B>
struct wedge_coeffwise_extents_values<extents<A...>, extents<B...>> {
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

        constexpr auto ADI = AEI::dynamic_indices;
        constexpr auto BDI = BEI::dynamic_indices;

        return product_extents_type(a.extent(ADI[N])..., b.extent(BDI[M])...);
    }

    constexpr static product_extents_type merge(const extents<A...>& a,
                                                const extents<B...>& b) {
        return merge(
            a, b,
            std::make_integer_sequence<rank_type, rank_type(sizeof...(A))>{},
            std::make_integer_sequence<rank_type, rank_type(sizeof...(B))>{});
    }
};

}  // namespace _detail_wedge

template <concepts::Expression A, concepts::Expression B>
struct detail::ExpressionTraits<binary::WedgeProduct<A, B>>
    : public binary::detail::DefaultBinaryExpressionTraits<A, B> {
    using ATraits = detail::ExpressionTraits<A>;
    constexpr static rank_type lhs_rank = ATraits::extents_type::rank();
    using BTraits = detail::ExpressionTraits<B>;
    constexpr static rank_type rhs_rank = BTraits::extents_type::rank();
    using CEV =
        _detail_wedge::wedge_coeffwise_extents_values<typename ATraits::extents_type,
                                               typename BTraits::extents_type>;

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
template <zipper::concepts::Expression A, zipper::concepts::Expression B>
class WedgeProduct : public BinaryExpressionBase<WedgeProduct<A, B>, const A, const B> {
   public:
    using self_type = WedgeProduct<A, B>;
    using Base = BinaryExpressionBase<self_type, const A, const B>;
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

    WedgeProduct(const A& a, const B& b)
        requires(extents_traits::is_static)
        : Base(a, b) {}
    WedgeProduct(const A& a, const B& b)
        requires(!extents_traits::is_static)
        : Base(a, b, traits::CEV::merge(a.extents(), b.extents())) {}

    template <bool DoOffset, typename... Args, rank_type... ranks>
    auto lhs_value(std::integer_sequence<rank_type, ranks...>,
                   Args&&... args) const -> decltype(auto) {
        return lhs()(zipper::detail::pack_index<ranks + (DoOffset ? rhs_rank : 0)>(
            std::forward<Args>(args)...)...);
    }
    template <bool DoOffset, typename... Args, rank_type... ranks>
    auto rhs_value(std::integer_sequence<rank_type, ranks...>,
                   Args&&... args) const -> decltype(auto) {
        return rhs()(zipper::detail::pack_index<ranks + (DoOffset ? lhs_rank : 0)>(
            std::forward<Args>(args)...)...);
    }

    template <typename... Args>
    value_type coeff(Args&&... args) const {
        return lhs_value<false>(
                   std::make_integer_sequence<rank_type, lhs_rank>{},
                   std::forward<Args>(args)...) *
                   rhs_value<true>(
                       std::make_integer_sequence<rank_type, rhs_rank>{},
                       std::forward<Args>(args)...)

               - lhs_value<true>(
                     std::make_integer_sequence<rank_type, lhs_rank>{},
                     std::forward<Args>(args)...) *
                     rhs_value<false>(
                         std::make_integer_sequence<rank_type, rhs_rank>{},
                         std::forward<Args>(args)...);
    }
};

template <zipper::concepts::Expression A, zipper::concepts::Expression B>
WedgeProduct(const A& a, const B& b) -> WedgeProduct<A, B>;

}  // namespace binary
}  // namespace zipper::expression
#endif
