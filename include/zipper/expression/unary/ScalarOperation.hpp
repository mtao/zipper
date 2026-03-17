#if !defined(ZIPPER_expression_UNARY_SCALAROPERATIONVIEW_HPP)
#define ZIPPER_expression_UNARY_SCALAROPERATIONVIEW_HPP

#include "UnaryExpressionBase.hpp"
#include "zipper/expression/detail/IndexSet.hpp"

namespace zipper::expression {
namespace unary {
template <zipper::concepts::QualifiedExpression Child, typename Operation,
          typename Scalar, bool ScalarOnRight = false>
class ScalarOperation;

}
template <zipper::concepts::QualifiedExpression Child, typename Operation,
          typename Scalar, bool ScalarOnRight>
struct detail::ExpressionTraits<
    unary::ScalarOperation<Child, Operation, Scalar, ScalarOnRight>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
          Child,
          zipper::detail::AccessFeatures{.is_const = true,
                                         .is_reference = false}> {
  using ChildTraits = ExpressionTraits<std::decay_t<Child>>;
  using value_type = decltype(std::declval<Operation>()(
      std::declval<typename ChildTraits::value_type>(),
      std::declval<Scalar>()));

  /// Propagate has_index_set when the scalar Op preserves zeros.
  constexpr static bool has_index_set =
      ChildTraits::has_index_set &&
      zipper::expression::detail::ZeroPreservingScalarOp<Operation,
                                                         ScalarOnRight>;

  /// Backward-compatible alias for has_index_set.
  constexpr static bool has_known_zeros = has_index_set;
};

namespace unary {
template <zipper::concepts::QualifiedExpression Child, typename Operation,
          typename Scalar, bool ScalarOnRight>
class ScalarOperation
    : public UnaryExpressionBase<
          ScalarOperation<Child, Operation, Scalar, ScalarOnRight>, Child> {
public:
  using self_type = ScalarOperation<Child, Operation, Scalar, ScalarOnRight>;
  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using extents_type = traits::extents_type;
  using value_type = traits::value_type;

  using Base = UnaryExpressionBase<self_type, Child>;
  using Base::expression;

  template <typename U>
    requires (ScalarOnRight && std::constructible_from<typename Base::storage_type, U &&>)
  ScalarOperation(U &&a, const Scalar &b, const Operation &op = {})
      : Base(std::forward<U>(a)), m_op(op), m_scalar(b) {}
  template <typename U>
    requires (!ScalarOnRight && std::constructible_from<typename Base::storage_type, U &&>)
  ScalarOperation(const Scalar &a, U &&b, const Operation &op = {})
      : Base(std::forward<U>(b)), m_op(op), m_scalar(a) {}

  // using child_value_type = traits::base_value_type;

  auto get_value(const auto &value) const -> value_type {
    if constexpr (ScalarOnRight) {
      return m_op(value, m_scalar);
    } else {
      return m_op(m_scalar, value);
    }
  }

  /// Recursively deep-copy child so the result owns all data.
  auto make_owned() const {
      auto owned_child = expression().make_owned();
      if constexpr (ScalarOnRight) {
          return ScalarOperation<const decltype(owned_child), Operation, Scalar, ScalarOnRight>(
              std::move(owned_child), m_scalar, m_op);
      } else {
          return ScalarOperation<const decltype(owned_child), Operation, Scalar, ScalarOnRight>(
              m_scalar, std::move(owned_child), m_op);
      }
  }

  // ── Index set forwarding ────────────────────────────────────────────
  // Zero-preserving scalar ops (e.g. multiplies, divides-on-right)
  // don't change the sparsity pattern.

  template <rank_type D, typename... Args>
    requires(traits::has_index_set)
  auto index_set(Args &&...args) const {
    return expression().template index_set<D>(
        std::forward<Args>(args)...);
  }

  /// @deprecated Use index_set instead.
  template <rank_type D, typename... Args>
    requires(traits::has_index_set)
  auto nonzero_range(Args &&...args) const {
    return index_set<D>(std::forward<Args>(args)...);
  }

  auto col_range_for_row(index_type row) const
    requires(traits::has_index_set && extents_type::rank() == 2)
  {
    return expression().col_range_for_row(row);
  }

  auto row_range_for_col(index_type col) const
    requires(traits::has_index_set && extents_type::rank() == 2)
  {
    return expression().row_range_for_col(col);
  }

  auto nonzero_segment() const
    requires(traits::has_index_set && extents_type::rank() == 1)
  {
    return expression().nonzero_segment();
  }

private:
  Operation m_op;
  Scalar m_scalar;
};

template <zipper::concepts::Expression Child, typename Operation,
          typename Scalar>
ScalarOperation(const Child &a, const Scalar &b, const Operation &op)
    -> ScalarOperation<const Child&, Operation, Scalar, true>;
template <zipper::concepts::Expression Child, typename Operation,
          typename Scalar>
ScalarOperation(const Scalar &a, const Child &b, const Operation &op)
    -> ScalarOperation<const Child&, Operation, Scalar, false>;
} // namespace unary
} // namespace zipper::expression
#endif
