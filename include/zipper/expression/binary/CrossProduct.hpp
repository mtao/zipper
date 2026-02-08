#if !defined(ZIPPER_EXPRESSION_BINARY_CROSSPRODUCT_HPP)
#define ZIPPER_EXPRESSION_BINARY_CROSSPRODUCT_HPP

#include "BinaryExpressionBase.hpp"
#include "detail/CoeffWiseTraits.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/detail/intersect_nonzeros.hpp"

namespace zipper::expression {
namespace binary {
template <zipper::concepts::RankedExpression<1> A,
          zipper::concepts::RankedExpression<1> B>
class CrossProduct;

}
template <concepts::RankedExpression<1> A, concepts::RankedExpression<1> B>
struct detail::ExpressionTraits<binary::CrossProduct<A, B>>
    : public binary::detail::DefaultBinaryExpressionTraits<A, B> {
    using ATraits = detail::ExpressionTraits<A>;
    using BTraits = detail::ExpressionTraits<B>;
    using ConvertExtentsUtil = binary::detail::coeffwise_extents_values<
        typename ATraits::extents_type, typename BTraits::extents_type>;
    using extents_type = typename ConvertExtentsUtil::merged_extents_type;

    // constraint on the extent, supporting 2d is ugly but who cares
    static_assert(extents_type::static_extent(0) == std::dynamic_extent ||
                  extents_type::static_extent(0) == 3);
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
};

namespace binary {
template <zipper::concepts::RankedExpression<1> A,
          zipper::concepts::RankedExpression<1> B>
class CrossProduct : public BinaryExpressionBase<CrossProduct<A, B>, const A, const B> {
   public:
    using self_type = CrossProduct<A, B>;
    using Base = BinaryExpressionBase<self_type, const A, const B>;
    using ExpressionBase = expression::ExpressionBase<self_type>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using value_type = traits::value_type;
    using extents_type = typename traits::extents_type;
    using lhs_traits = traits::ATraits;
    using rhs_traits = traits::BTraits;

    using Base::Base;
    using Base::lhs;
    using Base::rhs;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    CrossProduct(const A& a, const B& b)
        : Base(a, b) {
        assert(a.extent(0) == b.extent(0));
        assert(a.extent(0) == 3);
    }

    constexpr auto extent(rank_type i) const -> index_type {
        assert(i == 0);
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

};

template <zipper::concepts::RankedExpression<1> A,
          zipper::concepts::RankedExpression<1> B>
CrossProduct(const A& a, const B& b) -> CrossProduct<A, B>;

}  // namespace binary
}  // namespace zipper::expression
#endif
