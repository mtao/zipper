#if !defined(ZIPPER_EXPRESSION_BINARY_CROSSPRODUCT_HPP)
#define ZIPPER_EXPRESSION_BINARY_CROSSPRODUCT_HPP

#include "BinaryExpressionBase.hpp"
#include "detail/CoeffWiseTraits.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/assert.hpp"
#include "zipper/expression/detail/intersect_nonzeros.hpp"

namespace zipper::expression {
namespace binary {
template <zipper::concepts::QualifiedRankedExpression<1> A,
          zipper::concepts::QualifiedRankedExpression<1> B>
class CrossProduct;

}
template <concepts::QualifiedRankedExpression<1> A, concepts::QualifiedRankedExpression<1> B>
  requires(binary::detail::coeffwise_extents_values<
               typename detail::ExpressionTraits<std::decay_t<A>>::extents_type,
               typename detail::ExpressionTraits<std::decay_t<B>>::extents_type
           >::merged_extents_type::static_extent(0) == std::dynamic_extent ||
           binary::detail::coeffwise_extents_values<
               typename detail::ExpressionTraits<std::decay_t<A>>::extents_type,
               typename detail::ExpressionTraits<std::decay_t<B>>::extents_type
           >::merged_extents_type::static_extent(0) == 3)
struct detail::ExpressionTraits<binary::CrossProduct<A, B>>
    : public binary::detail::DefaultBinaryExpressionTraits<A, B> {
    using ATraits = detail::ExpressionTraits<std::decay_t<A>>;
    using BTraits = detail::ExpressionTraits<std::decay_t<B>>;
    using ConvertExtentsUtil = binary::detail::coeffwise_extents_values<
        typename ATraits::extents_type, typename BTraits::extents_type>;
    using extents_type = typename ConvertExtentsUtil::merged_extents_type;

    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
};

namespace binary {
template <zipper::concepts::QualifiedRankedExpression<1> A,
          zipper::concepts::QualifiedRankedExpression<1> B>
class CrossProduct : public BinaryExpressionBase<CrossProduct<A, B>, A, B> {
   public:
    using self_type = CrossProduct<A, B>;
    using Base = BinaryExpressionBase<self_type, A, B>;
    using ExpressionBase = expression::ExpressionBase<self_type>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using value_type = traits::value_type;
    using extents_type = typename traits::extents_type;
    using lhs_traits = traits::ATraits;
    using rhs_traits = traits::BTraits;

    using Base::lhs;
    using Base::rhs;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    CrossProduct(const CrossProduct&) = default;
    CrossProduct(CrossProduct&&) = default;

    template <typename U, typename V>
      requires std::constructible_from<typename Base::lhs_storage_type, U&&> &&
               std::constructible_from<typename Base::rhs_storage_type, V&&>
    CrossProduct(U&& a, V&& b)
        : Base(std::forward<U>(a), std::forward<V>(b)) {
        ZIPPER_ASSERT(lhs().extent(0) == rhs().extent(0));
        ZIPPER_ASSERT(lhs().extent(0) == 3);
    }

    constexpr auto extent([[maybe_unused]] rank_type i) const -> index_type {
        ZIPPER_ASSERT(i == 0);
        return lhs().extent(0);
    }

    constexpr auto extents() const -> extents_type {
        return extents_traits::make_extents_from(*this);
    }
    value_type coeff(index_type a) const {
        rank_type b = (a + 1) % 3;
        rank_type c = (a + 2) % 3;
        return lhs()(b) * rhs()(c) - lhs()(c) * rhs()(b);
    }

    /// Recursively deep-copy children so the result owns all data.
    auto make_owned() const {
        auto owned_a = lhs().make_owned();
        auto owned_b = rhs().make_owned();
        return CrossProduct<decltype(owned_a), decltype(owned_b)>(
            std::move(owned_a), std::move(owned_b));
    }

};

template <zipper::concepts::RankedExpression<1> A,
          zipper::concepts::RankedExpression<1> B>
CrossProduct(const A& a, const B& b) -> CrossProduct<const A&, const B&>;

}  // namespace binary
}  // namespace zipper::expression
#endif
