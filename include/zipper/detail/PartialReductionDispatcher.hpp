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
    DerivedT a = m_dispatcher.sum();
    return a;
  }
  template <index_type P = 2> auto norm() const {
    DerivedT a = m_dispatcher.norm();
    return a;
  }
  template <index_type P = 2> auto norm_powered() const {
    DerivedT a = m_dispatcher.norm_powered();
    return a;
  }

private:
  expression::unary::detail::PartialReductionDispatcher<ExprType, Indices...>
      m_dispatcher;
};

} // namespace zipper::detail
#endif
