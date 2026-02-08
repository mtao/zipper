
#if !defined(ZIPPER_EXPRESSION_UNARY_COFACTOR_HPP)
#define ZIPPER_EXPRESSION_UNARY_COFACTOR_HPP

#include "Slice.hpp"
#include "UnaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/reductions/Determinant.hpp"

#include <array>
#include <vector>

namespace zipper::expression {
namespace unary {

template <zipper::concepts::QualifiedExpression ExprType>
  requires(ExprType::extents_type::rank() == 2)
class Cofactor;

} // namespace unary

// ── Traits specialization ──────────────────────────────────────────────
template <zipper::concepts::QualifiedExpression ExprType>
  requires(ExprType::extents_type::rank() == 2)
struct detail::ExpressionTraits<unary::Cofactor<ExprType>>
    : public unary::detail::DefaultUnaryExpressionTraits<ExprType> {
  using child_traits = detail::ExpressionTraits<ExprType>;
  using value_type = typename child_traits::value_type;
  using extents_type = typename child_traits::extents_type;

  constexpr static bool is_coefficient_consistent = false;
  constexpr static bool is_value_based = false;

  constexpr static zipper::detail::AccessFeatures access_features = {
      .is_const = true,
      .is_reference = false,
      .is_alias_free = child_traits::access_features.is_alias_free,
  };
  consteval static auto is_const_valued() -> bool {
    return access_features.is_const;
  }
  consteval static auto is_reference_valued() -> bool {
    return access_features.is_reference;
  }
  consteval static auto is_assignable() -> bool {
    return !is_const_valued() && is_reference_valued();
  }
  consteval static auto is_referrable() -> bool {
    return access_features.is_reference;
  }
  constexpr static bool is_writable = is_assignable();
};

// ── Class definition ───────────────────────────────────────────────────
namespace unary {

template <zipper::concepts::QualifiedExpression ExprType>
  requires(ExprType::extents_type::rank() == 2)
class Cofactor
    : public UnaryExpressionBase<Cofactor<ExprType>, ExprType> {
public:
  using self_type = Cofactor<ExprType>;
  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using extents_type = typename traits::extents_type;
  using value_type = typename traits::value_type;
  using Base = UnaryExpressionBase<self_type, ExprType>;
  using Base::expression;

  Cofactor() = delete;
  Cofactor(const Cofactor &) = default;
  Cofactor(Cofactor &&) = default;
  auto operator=(const Cofactor &) -> Cofactor & = delete;
  auto operator=(Cofactor &&) -> Cofactor & = delete;

  Cofactor(ExprType &b) : Base(b) {
    assert(expression().extent(0) == expression().extent(1));
  }

  /// C(i,j) = (-1)^(i+j) * det(minor(M, i, j))
  auto coeff(index_type i, index_type j) const -> value_type {
    const index_type n = expression().extent(0);
    const value_type sign =
        ((i + j) % 2 == 0) ? value_type(1) : value_type(-1);

    if (n == 1) {
      return sign;
    }

    constexpr auto static_n = extents_type::static_extent(0);
    if constexpr (static_n != std::dynamic_extent) {
      // Static path: use std::array to avoid heap allocations
      std::array<index_type, static_n - 1> rows;
      std::array<index_type, static_n - 1> cols;
      index_type ri = 0, ci = 0;
      for (index_type r = 0; r < static_n; ++r) {
        if (r != i) rows[ri++] = r;
      }
      for (index_type c = 0; c < static_n; ++c) {
        if (c != j) cols[ci++] = c;
      }
      auto minor_expr = Slice(expression(), rows, cols);
      return sign * reductions::Determinant(minor_expr)();
    } else {
      // Dynamic path: use std::vector
      std::vector<index_type> rows;
      std::vector<index_type> cols;
      rows.reserve(n - 1);
      cols.reserve(n - 1);
      for (index_type r = 0; r < n; ++r) {
        if (r != i) rows.push_back(r);
      }
      for (index_type c = 0; c < n; ++c) {
        if (c != j) cols.push_back(c);
      }
      auto minor_expr = Slice(expression(), rows, cols);
      return sign * reductions::Determinant(minor_expr)();
    }
  }
};

template <zipper::concepts::QualifiedExpression ExprType>
Cofactor(ExprType &) -> Cofactor<ExprType>;

} // namespace unary
} // namespace zipper::expression
#endif
