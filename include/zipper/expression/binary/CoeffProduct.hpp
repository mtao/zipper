#if !defined(ZIPPER_EXPRESSION_BINARY_COEFFPRODUCT_HPP)
#define ZIPPER_EXPRESSION_BINARY_COEFFPRODUCT_HPP

#include "BinaryExpressionBase.hpp"
#include "detail/CoeffWiseTraits.hpp"
#include "zipper/concepts/Expression.hpp"

namespace zipper::expression {
namespace binary {
template <zipper::concepts::Expression A, zipper::concepts::Expression B>
class CoeffProduct;

}
template <concepts::Expression A, concepts::Expression B>
struct detail::ExpressionTraits<binary::CoeffProduct<A, B>>
    : public binary::detail::DefaultBinaryExpressionTraits<A, B> {
    using Base = detail::ExpressionTraits<A>;
    using extents_type = typename Base::extents_type;
};

namespace binary {
// Per-coefficient product (i.e A(x,y,z) * B(x,y,z))
template <zipper::concepts::Expression A, zipper::concepts::Expression B>
class CoeffProduct : public BinaryExpressionBase<CoeffProduct<A, B>, const A, const B> {
   public:
    using self_type = CoeffProduct<A, B>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using Base = BinaryExpressionBase<self_type, const A, const B>;
    using Base::Base;
    using Base::lhs;
    using Base::rhs;
    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    constexpr auto extent(rank_type i) const -> index_type {
        return lhs().extent(i);
    }

    constexpr auto extents() const -> extents_type {
        return extents_traits::make_extents_from(*this);
    }

    template <typename... Args>
    auto coeff(Args&&... idxs) const {
        return lhs()(idxs...) * rhs()(idxs...);
    }
};

template <zipper::concepts::Expression A, zipper::concepts::Expression B>
CoeffProduct(const A& a, const B& b) -> CoeffProduct<A, B>;

}  // namespace binary
}  // namespace zipper::expression
#endif
