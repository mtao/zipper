#if !defined(ZIPPER_EXPRESSION_BINARY_ADDITION_HPP)
#define ZIPPER_EXPRESSION_BINARY_ADDITION_HPP

#include "BinaryExpressionBase.hpp"
#include "detail/CoeffWiseTraits.hpp"
#include "zipper/concepts/Index.hpp"
#include "zipper/concepts/Expression.hpp"

namespace zipper::expression {
namespace binary {
template <zipper::concepts::Expression A, zipper::concepts::Expression B>
class Addition;

}
template <concepts::Expression A, concepts::Expression B>
struct detail::ExpressionTraits<binary::Addition<A, B>>
    : public binary::detail::DefaultBinaryExpressionTraits<A, B> {
    using ATraits = detail::ExpressionTraits<A>;
    using BTraits = detail::ExpressionTraits<B>;
    using ConvertExtentsUtil = binary::detail::coeffwise_extents_values<
        typename ATraits::extents_type, typename BTraits::extents_type>;
    using extents_type = typename ConvertExtentsUtil::merged_extents_type;
};

namespace binary {
// Per-coefficient addition (i.e A(x,y,z) + B(x,y,z))
template <zipper::concepts::Expression A, zipper::concepts::Expression B>
class Addition : public BinaryExpressionBase<Addition<A, B>, const A, const B> {
   public:
    using self_type = Addition<A, B>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using extents_type = typename traits::extents_type;
    static_assert(A::extents_type::rank() == extents_type::rank());
    static_assert(B::extents_type::rank() == extents_type::rank());
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    using Base = BinaryExpressionBase<self_type, const A, const B>;

    using Base::Base;
    using Base::lhs;
    using Base::rhs;
    Addition(const A& a, const B& b)
        requires(!extents_traits::is_static)
        : Base(a, b, a.extents()) {}

    template <typename... Args>
        requires concepts::IndexPack<Args...>
    auto coeff(Args&&... idxs) const {
        static_assert(sizeof...(idxs) == extents_type::rank());
        return lhs()(idxs...) + rhs()(idxs...);
    }
};

template <zipper::concepts::Expression A, zipper::concepts::Expression B>
Addition(const A& a, const B& b) -> Addition<A, B>;

}  // namespace binary
}  // namespace zipper::expression
#endif
