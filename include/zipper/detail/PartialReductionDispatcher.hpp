#if !defined(ZIPPER_DETAIL_PARTIALREDUCTIONDISPATCHER_HPP)
#define ZIPPER_DETAIL_PARTIALREDUCTIONDISPATCHER_HPP
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/wrap_in_base.hpp"
#include "zipper/expression/reductions/CoefficientMinMax.hpp"
#include "zipper/expression/unary/detail/PartialReductionDispatcher.hpp"

#include <utility>

namespace zipper::detail {

/// Wraps partial reductions in a semantic base (DerivedT).
///
/// ExprType is the child storage type computed by member_child_storage_t:
/// a value when called on a temporary (the result then owns the child), an
/// lvalue reference otherwise.
template <template <typename> typename DerivedT,
          zipper::concepts::QualifiedExpression ExprType,
          rank_type... Indices>
class PartialReductionDispatcher {
    using inner_type =
        expression::unary::detail::PartialReductionDispatcher<ExprType,
                                                              Indices...>;

  public:
    template <typename U>
        requires std::constructible_from<ExprType, U &&>
    explicit PartialReductionDispatcher(U &&v)
      : m_dispatcher(std::forward<U>(v)) {}

    template <template <typename> typename Reduction, typename Self>
    auto reduce(this Self &&self) {
        using PR =
            expression::unary::PartialReduction<ExprType, Reduction, Indices...>;
        return DerivedT<PR>(std::in_place,
                            std::forward<Self>(self)
                                .m_dispatcher.forward_expression());
    }

    template <typename Self>
    auto sum(this Self &&self) {
        return std::forward<Self>(self)
            .template reduce<expression::reductions::CoefficientSum>();
    }
    /// Per-slice minimum / maximum coefficient (slices must be non-empty).
    template <typename Self>
    auto min(this Self &&self) {
        using holder = expression::reductions::detail::
            coefficient_extremum_holder<std::less<>>;
        return std::forward<Self>(self)
            .template reduce<holder::template CoefficientExtremum>();
    }
    template <typename Self>
    auto max(this Self &&self) {
        using holder = expression::reductions::detail::
            coefficient_extremum_holder<std::greater<>>;
        return std::forward<Self>(self)
            .template reduce<holder::template CoefficientExtremum>();
    }
    template <index_type P = 2, typename Self>
    auto norm(this Self &&self) {
        using holder = expression::reductions::detail::lp_norm_holder<P>;
        return std::forward<Self>(self).template reduce<holder::template LpNorm>();
    }
    template <index_type P = 2, typename Self>
    auto norm_powered(this Self &&self) {
        using holder =
            expression::reductions::detail::lp_norm_powered_holder<P>;
        return std::forward<Self>(self)
            .template reduce<holder::template LpNormPowered>();
    }

    template <typename F, typename Self>
    auto transform(this Self &&self, F &&fn) {
        using PT = expression::unary::
            PartialTransform<ExprType, std::decay_t<F>, Indices...>;
        // PartialTransform preserves rank (unlike PartialReduction which
        // reduces it), so wrap based on the actual output rank, not DerivedT.
        return wrap_in_base<PT>(
            std::forward<Self>(self).m_dispatcher.forward_expression(),
            std::forward<F>(fn));
    }

  private:
    inner_type m_dispatcher;
};

} // namespace zipper::detail
#endif
