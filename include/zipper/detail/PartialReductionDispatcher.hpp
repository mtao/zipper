#if !defined(ZIPPER_DETAIL_PARTIALREDUCTIONDISPATCHER_HPP)
#define ZIPPER_DETAIL_PARTIALREDUCTIONDISPATCHER_HPP
#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/unary/detail/PartialReductionDispatcher.hpp"

namespace zipper::detail {

template <template <concepts::QualifiedExpression> typename DerivedT,
          zipper::concepts::QualifiedExpression ExprType, rank_type... Indices>
class PartialReductionDispatcher {
public:
  PartialReductionDispatcher(ExprType &v) : m_dispatcher(v) {}
  auto sum() const {
    using PR = expression::unary::PartialReduction<
        ExprType&, expression::reductions::CoefficientSum, Indices...>;
    return DerivedT<PR>(std::in_place, m_dispatcher.expression());
  }
  template <index_type P = 2> auto norm() const {
    using holder = expression::reductions::detail::lp_norm_holder<P>;
    using PR = expression::unary::PartialReduction<
        ExprType&, holder::template LpNorm, Indices...>;
    return DerivedT<PR>(std::in_place, m_dispatcher.expression());
  }
  template <index_type P = 2> auto norm_powered() const {
    using holder = expression::reductions::detail::lp_norm_powered_holder<P>;
    using PR = expression::unary::PartialReduction<
        ExprType&, holder::template LpNormPowered, Indices...>;
    return DerivedT<PR>(std::in_place, m_dispatcher.expression());
  }

private:
  expression::unary::detail::PartialReductionDispatcher<ExprType, Indices...>
      m_dispatcher;
};

} // namespace zipper::detail
#endif
