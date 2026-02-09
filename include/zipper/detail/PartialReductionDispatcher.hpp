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
    auto expr = m_dispatcher.sum();
    DerivedT<decltype(expr)> a(std::move(expr));
    return a;
  }
  template <index_type P = 2> auto norm() const {
    auto expr = m_dispatcher.template norm<P>();
    DerivedT<decltype(expr)> a(std::move(expr));
    return a;
  }
  template <index_type P = 2> auto norm_powered() const {
    auto expr = m_dispatcher.template norm_powered<P>();
    DerivedT<decltype(expr)> a(std::move(expr));
    return a;
  }

private:
  expression::unary::detail::PartialReductionDispatcher<ExprType, Indices...>
      m_dispatcher;
};

} // namespace zipper::detail
#endif
