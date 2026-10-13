#if !defined(ZIPPER_EXPRESSION_UNARY_DETAIL_PARTIALREDUCTIONDISPATCHER_HPP)
#define ZIPPER_EXPRESSION_UNARY_DETAIL_PARTIALREDUCTIONDISPATCHER_HPP
#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/reductions/CoefficientSum.hpp"
#include "zipper/expression/reductions/LpNorm.hpp"
#include "zipper/expression/reductions/LpNormPowered.hpp"
#include "zipper/expression/unary/PartialReduction.hpp"
#include "zipper/expression/unary/PartialTransform.hpp"

#include <utility>

namespace zipper::expression::unary::detail {

/// Builds partial reductions / transforms over the child expression.
///
/// ExprType is the child storage type: a value type when the dispatcher owns
/// the child (e.g. it was created from a temporary), or an lvalue reference
/// when it borrows it. The produced expressions use the same storage type, so
/// an owned child is copied/moved into the result instead of being referenced
/// (which would dangle once the temporary is destroyed).
template <zipper::concepts::QualifiedExpression ExprType, rank_type... Indices>
class PartialReductionDispatcher {
  public:
    template <typename U>
        requires std::constructible_from<ExprType, U &&>
    explicit PartialReductionDispatcher(U &&v)
      : m_expression(std::forward<U>(v)) {}

    auto expression() const -> const std::remove_reference_t<ExprType> & {
        return m_expression;
    }

    /// Forwards the stored child: moves an owned child out of an rvalue
    /// dispatcher, otherwise yields an lvalue (copy source or reference).
    template <typename Self>
    auto forward_expression(this Self &&self) -> decltype(auto) {
        if constexpr (std::is_reference_v<ExprType>) {
            return static_cast<ExprType>(self.m_expression);
        } else {
            return (std::forward<Self>(self).m_expression);
        }
    }

    template <template <typename> typename Reduction, typename Self>
    auto reduce(this Self &&self) {
        return unary::PartialReduction<ExprType, Reduction, Indices...>(
            std::forward<Self>(self).forward_expression());
    }

    template <typename Self>
    auto sum(this Self &&self) {
        return std::forward<Self>(self)
            .template reduce<reductions::CoefficientSum>();
    }
    template <index_type P = 2, typename Self>
    auto norm(this Self &&self) {
        using holder = reductions::detail::lp_norm_holder<P>;
        return std::forward<Self>(self).template reduce<holder::template LpNorm>();
    }
    template <index_type P = 2, typename Self>
    auto norm_powered(this Self &&self) {
        using holder = reductions::detail::lp_norm_powered_holder<P>;
        return std::forward<Self>(self)
            .template reduce<holder::template LpNormPowered>();
    }

    template <typename F, typename Self>
    auto transform(this Self &&self, F &&fn) {
        return unary::PartialTransform<ExprType, std::decay_t<F>, Indices...>(
            std::forward<Self>(self).forward_expression(), std::forward<F>(fn));
    }

  private:
    ExprType m_expression;
};

} // namespace zipper::expression::unary::detail

#endif
