#if !defined(ZIPPER_EXPRESSION_UNARY_DETAIL_PARTIALREDUCTIONDISPATCHER_HPP)
#define ZIPPER_EXPRESSION_UNARY_DETAIL_PARTIALREDUCTIONDISPATCHER_HPP

#include "zipper/concepts/Expression.hpp"
#include "zipper/concepts/Zipper.hpp"
#include "zipper/expression/reductions/CoefficientMinMax.hpp"
#include "zipper/expression/reductions/CoefficientSum.hpp"
#include "zipper/expression/reductions/InnerProduct.hpp"
#include "zipper/expression/reductions/LpNorm.hpp"
#include "zipper/expression/reductions/LpNormPowered.hpp"
#include "zipper/expression/unary/PartialMap.hpp"
#include "zipper/expression/unary/detail/ReduceWith.hpp"

#include <functional>
#include <optional>
#include <type_traits>
#include <utility>

namespace zipper::expression::unary::detail {

/// Operations on the sub-tensors of a child spanning axes Indices...
/// (zipper's over<Indices...>(), colwise(), rowwise()): each returns a raw
/// PartialMap expression; wrap it (as_array(...), zipper::Vector(...), ...)
/// for further operations.
///
/// ExprType is the child storage type: a value type when the dispatcher owns
/// the child (e.g. it was created from a temporary), or an lvalue reference
/// when it borrows it. The produced expressions use the same storage type, so
/// an owned child is copied/moved into the result instead of being referenced
/// (which would dangle once the temporary is destroyed).
template <zipper::concepts::QualifiedExpression ExprType, rank_type... Indices>
class PartialReductionDispatcher {
    template <typename F, rank_type... Out>
    using map_type =
        unary::PartialMap<ExprType, std::decay_t<F>, std::integer_sequence<rank_type, Out...>,
                          Indices...>;

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

    /// Maps each sub-tensor (a raw expression) with `fn`, which returns a
    /// scalar, an expression or zipper object, or a callable taking the
    /// sub-tensor's indices; the axes of its result go to positions Out...
    /// of the result (default: none for scalars, in place for results
    /// shaped like the sub-tensors).
    ///
    ///     rgb.over<0>().map<2>(to_rgba)   // (3, W, H) -> (W, H, 4)
    template <rank_type... Out, typename F, typename Self>
    auto map(this Self &&self, F &&fn) {
        return map_type<F, Out...>(std::forward<Self>(self).forward_expression(),
                                   std::forward<F>(fn));
    }
    /// As map(fn), with the extents of fn's results given rather than
    /// taken from fn applied to the first sub-tensor (when not static).
    template <rank_type... Out, typename F, typename Self>
    auto map(this Self &&self,
             F &&fn,
             const typename expression::detail::ExpressionDetail<
                 map_type<F, Out...>>::result_extents_type &result_extents) {
        return map_type<F, Out...>(std::forward<Self>(self).forward_expression(),
                                   std::forward<F>(fn), std::optional(result_extents));
    }

    /// Maps each sub-tensor to a tensor of the same shape, in place.
    template <typename F, typename Self>
    auto transform(this Self &&self, F &&fn) {
        using detail_type = expression::detail::ExpressionDetail<map_type<F>>;
        static_assert(!detail_type::scalar_result,
                      "transform maps sub-tensors to tensors; use map for scalars");
        // Same shape: no need to evaluate fn to find it.
        const typename detail_type::result_extents_type extents(
            self.m_expression.extent(Indices)...);
        return std::forward<Self>(self).map(std::forward<F>(fn), extents);
    }

    /// Reduces each sub-tensor with Reduction, constructed with `parameters`.
    template <template <typename> typename Reduction, typename Self, typename... Parameters>
    auto reduce(this Self &&self, Parameters &&...parameters) {
        return std::forward<Self>(self).map(ReduceWith<Reduction, std::decay_t<Parameters>...>{
            {std::forward<Parameters>(parameters)...}});
    }

    template <typename Self>
    auto sum(this Self &&self) {
        return std::forward<Self>(self).template reduce<reductions::CoefficientSum>();
    }
    /// Per-sub-tensor minimum / maximum coefficient (sub-tensors must be
    /// non-empty).
    template <typename Self>
    auto min(this Self &&self) {
        using holder = reductions::detail::coefficient_extremum_holder<std::less<>>;
        return std::forward<Self>(self).template reduce<holder::template CoefficientExtremum>();
    }
    template <typename Self>
    auto max(this Self &&self) {
        using holder = reductions::detail::coefficient_extremum_holder<std::greater<>>;
        return std::forward<Self>(self).template reduce<holder::template CoefficientExtremum>();
    }
    template <index_type P = 2, typename Self>
    auto norm(this Self &&self) {
        using holder = reductions::detail::lp_norm_holder<P>;
        return std::forward<Self>(self).template reduce<holder::template LpNorm>();
    }
    template <typename Self>
    auto norm(this Self &&self, typename std::decay_t<ExprType>::value_type p) {
        using value_type = typename std::decay_t<ExprType>::value_type;
        using binder = reductions::detail::lp_norm_binder<value_type>;
        return std::forward<Self>(self).template reduce<binder::template reduction>(p);
    }
    template <index_type P = 2, typename Self>
    auto norm_powered(this Self &&self) {
        using holder = reductions::detail::lp_norm_powered_holder<P>;
        return std::forward<Self>(self).template reduce<holder::template LpNormPowered>();
    }

    /// Contracts each sub-tensor with `other` (an expression or zipper object
    /// of the same shape): their inner product, e.g. per-pixel luminance
    /// from RGB weights.  A temporary `other` is owned by the result;
    /// otherwise it is referenced.
    template <typename Other, typename Self>
        requires(zipper::concepts::Zipper<Other> || zipper::concepts::Expression<Other>)
    auto contract(this Self &&self, Other &&other) {
        decltype(auto) other_expression = [&]() -> decltype(auto) {
            if constexpr (zipper::concepts::Zipper<Other>) {
                return std::forward<Other>(other).expression();
            } else {
                return std::forward<Other>(other);
            }
        }();
        using raw = std::decay_t<decltype(other_expression)>;
        using other_t =
            std::conditional_t<std::is_lvalue_reference_v<Other>, const raw &, raw>;
        using binder = reductions::detail::inner_product_binder<other_t>;
        return std::forward<Self>(self).map(ReduceWith<binder::template reduction, other_t>{
            {static_cast<other_t>(std::forward<decltype(other_expression)>(other_expression))}});
    }

  private:
    ExprType m_expression;
};

} // namespace zipper::expression::unary::detail
#endif
