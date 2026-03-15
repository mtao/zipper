#if !defined(ZIPPER_EXPRESSION_BINARY_WEDGEPRODUCT_HPP)
#define ZIPPER_EXPRESSION_BINARY_WEDGEPRODUCT_HPP

#include "BinaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/pack_index.hpp"

namespace zipper::expression {
namespace binary {
template <zipper::concepts::QualifiedExpression A, zipper::concepts::QualifiedExpression B>
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

/// Implementation details for WedgeProduct expressions.
///
/// Holds child traits aliases, child rank constants (lhs_rank, rhs_rank),
/// and the wedge-product-specific extents computation utility (CEV) that
/// concatenates the LHS and RHS extents packs.
template <zipper::concepts::QualifiedExpression A, zipper::concepts::QualifiedExpression B>
struct detail::ExpressionDetail<binary::WedgeProduct<A, B>>
    : public binary::detail::DefaultBinaryExpressionDetail<A, B> {
    using _Base = binary::detail::DefaultBinaryExpressionDetail<A, B>;
    using ATraits = typename _Base::ATraits;
    using BTraits = typename _Base::BTraits;
    constexpr static rank_type lhs_rank = ATraits::extents_type::rank();
    constexpr static rank_type rhs_rank = BTraits::extents_type::rank();
    using CEV =
        _detail_wedge::wedge_coeffwise_extents_values<typename ATraits::extents_type,
                                               typename BTraits::extents_type>;
};

template <zipper::concepts::QualifiedExpression A, zipper::concepts::QualifiedExpression B>
struct detail::ExpressionTraits<binary::WedgeProduct<A, B>>
    : public binary::detail::DefaultBinaryExpressionTraits<A, B> {
    using _Detail = detail::ExpressionDetail<binary::WedgeProduct<A, B>>;

    static_assert(std::is_same_v<typename _Detail::CEV::a_extents_type,
                                 typename _Detail::ATraits::extents_type>);
    static_assert(std::is_same_v<typename _Detail::CEV::b_extents_type,
                                 typename _Detail::BTraits::extents_type>);
    using extents_type = typename _Detail::CEV::product_extents_type;
    static_assert(extents_type::rank() == _Detail::lhs_rank + _Detail::rhs_rank);

    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
};

namespace binary {
template <zipper::concepts::QualifiedExpression A, zipper::concepts::QualifiedExpression B>
class WedgeProduct : public BinaryExpressionBase<WedgeProduct<A, B>, A, B> {
   public:
    using self_type = WedgeProduct<A, B>;
    using Base = BinaryExpressionBase<self_type, A, B>;
    using ExpressionBase = expression::ExpressionBase<self_type>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using detail_type = zipper::expression::detail::ExpressionDetail<self_type>;
    using value_type = traits::value_type;
    using Base::lhs;
    using Base::rhs;
    constexpr static rank_type lhs_rank = detail_type::lhs_rank;
    constexpr static rank_type rhs_rank = detail_type::rhs_rank;

    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    template <typename U, typename V>
      requires std::constructible_from<typename Base::lhs_storage_type, U&&> &&
               std::constructible_from<typename Base::rhs_storage_type, V&&>
    WedgeProduct(U&& a, V&& b)
        : Base(std::forward<U>(a), std::forward<V>(b)) {}

    constexpr auto extent(rank_type i) const -> index_type {
        if (i < lhs_rank) {
            return lhs().extent(i);
        } else {
            return rhs().extent(i - lhs_rank);
        }
    }

    constexpr auto extents() const -> extents_type {
        return extents_traits::make_extents_from(*this);
    }

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

    /// Recursively deep-copy children so the result owns all data.
    auto make_owned() const {
        auto owned_a = lhs().make_owned();
        auto owned_b = rhs().make_owned();
        return WedgeProduct<decltype(owned_a), decltype(owned_b)>(
            std::move(owned_a), std::move(owned_b));
    }

};

template <zipper::concepts::Expression A, zipper::concepts::Expression B>
WedgeProduct(const A& a, const B& b) -> WedgeProduct<const A&, const B&>;

}  // namespace binary
}  // namespace zipper::expression
#endif
