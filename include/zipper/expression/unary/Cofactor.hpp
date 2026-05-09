
#if !defined(ZIPPER_EXPRESSION_UNARY_COFACTOR_HPP)
#define ZIPPER_EXPRESSION_UNARY_COFACTOR_HPP

#include "Slice.hpp"
#include "UnaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/assert.hpp"
#include "zipper/expression/reductions/Determinant.hpp"

#include <array>
#include <vector>

namespace zipper::expression {
namespace unary {

template <zipper::concepts::QualifiedExpression ExprType>
  requires(std::decay_t<ExprType>::extents_type::rank() == 2)
class Cofactor;

} // namespace unary

// ── Traits specialization ──────────────────────────────────────────────
template <zipper::concepts::QualifiedExpression ExprType>
  requires(std::decay_t<ExprType>::extents_type::rank() == 2)
struct detail::ExpressionTraits<unary::Cofactor<ExprType>>
    : public unary::detail::DefaultUnaryExpressionTraits<
          ExprType,
          zipper::detail::AccessFeatures::const_value()> {
  using child_traits = detail::ExpressionTraits<std::decay_t<ExprType>>;
  using value_type = typename child_traits::value_type;
  using extents_type = typename child_traits::extents_type;

  constexpr static bool is_coefficient_consistent = false;
  constexpr static bool is_value_based = false;
};

// ── Class definition ───────────────────────────────────────────────────
namespace unary {

template <zipper::concepts::QualifiedExpression ExprType>
  requires(std::decay_t<ExprType>::extents_type::rank() == 2)
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

  template <typename U>
    requires std::constructible_from<typename Base::storage_type, U &&>
  Cofactor(U &&b) : Base(std::forward<U>(b)) {
    ZIPPER_ASSERT(expression().extent(0) == expression().extent(1));
  }

  /// Recursively deep-copy child so the result owns all data.
  auto make_owned() const {
      auto owned_child = expression().make_owned();
      return Cofactor<const decltype(owned_child)>(
          std::move(owned_child));
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

template <zipper::concepts::Expression ExprType>
Cofactor(ExprType &) -> Cofactor<ExprType&>;

template <zipper::concepts::Expression ExprType>
Cofactor(const ExprType &) -> Cofactor<const ExprType&>;

} // namespace unary
} // namespace zipper::expression
#endif
