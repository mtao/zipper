#pragma once

#include <algorithm>
#include <functional>

#include "ReductionBase.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"

namespace zipper::expression::reductions {

namespace detail {
/// Reduces all coefficients with a binary selector (std::less -> min,
/// std::greater -> max).  Precondition: the expression is non-empty.
template <typename Compare>
struct coefficient_extremum_holder {
    template <typename Expr>
    class CoefficientExtremum
      : public ReductionBase<CoefficientExtremum<Expr>, Expr> {
      public:
        using Base = ReductionBase<CoefficientExtremum<Expr>, Expr>;
        using typename Base::expression_traits;
        using element_type = typename expression_traits::element_type;
        using Base::Base;
        using Base::expression;

        auto operator()() const -> element_type {
            bool first = true;
            element_type v{};
            for (const auto &i : zipper::utils::extents::all_extents_indices(
                     expression().extents())) {
                element_type x = std::apply(expression(), i);
                if (first || Compare{}(x, v)) {
                    v = x;
                    first = false;
                }
            }
            ZIPPER_ASSERT(!first);
            return v;
        }
    };
};
} // namespace detail

template <typename Expr>
using CoefficientMin = typename detail::coefficient_extremum_holder<
    std::less<>>::template CoefficientExtremum<Expr>;
template <typename Expr>
using CoefficientMax = typename detail::coefficient_extremum_holder<
    std::greater<>>::template CoefficientExtremum<Expr>;

} // namespace zipper::expression::reductions
