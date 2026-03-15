#if !defined(ZIPPER_EXPRESSION_UNARY_UNSAFEREF_HPP)
#define ZIPPER_EXPRESSION_UNARY_UNSAFEREF_HPP

#include "UnaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"

namespace zipper::expression {
namespace unary {

/// UnsafeRef is a transparent identity wrapper whose sole purpose is to
/// override stores_references to false, making the expression copyable,
/// movable and returnable even when it (transitively) stores references.
///
/// This is *unsafe* because the caller is asserting that the referenced
/// data outlives all copies of this expression.  Prefer to_owned() when
/// the lifetime guarantee cannot be made.
///
/// Usage:
///   auto v = some_expr.unsafe();       // v can be returned / stored
template <zipper::concepts::QualifiedExpression Child>
class UnsafeRef;

} // namespace unary

// ── Traits specialization ──────────────────────────────────────────────
template <zipper::concepts::QualifiedExpression Child>
struct detail::ExpressionTraits<unary::UnsafeRef<Child>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
          Child> {
  using child_traits =
      detail::ExpressionTraits<std::decay_t<Child>>;

  // Inherit everything from DefaultUnaryExpressionTraits except:
  // 1. Force stores_references = false (the whole point of UnsafeRef)
  // 2. Set is_value_based = false so the wrapper provides its own
  //    coeff / coeff_ref / const_coeff_ref that forward directly
  //    to the child without going through get_value().
  constexpr static bool stores_references = false;
  constexpr static bool is_value_based = false;
  constexpr static bool is_coefficient_consistent = false;

  /// Propagate has_known_zeros from child — UnsafeRef is transparent.
  constexpr static bool has_known_zeros = child_traits::has_known_zeros;
};

// ── Class definition ───────────────────────────────────────────────────
namespace unary {

template <zipper::concepts::QualifiedExpression Child>
class UnsafeRef
    : public UnaryExpressionBase<UnsafeRef<Child>, Child> {
 public:
  using self_type = UnsafeRef<Child>;
  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using extents_type = typename traits::extents_type;
  using value_type = typename traits::value_type;
  using Base = UnaryExpressionBase<self_type, Child>;
  using Base::expression;
  using Base::extent;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

  UnsafeRef(const UnsafeRef &) = default;
  UnsafeRef(UnsafeRef &&) = default;
  auto operator=(const UnsafeRef &) -> UnsafeRef & = delete;
  auto operator=(UnsafeRef &&) -> UnsafeRef & = delete;

  template <typename U>
    requires std::constructible_from<typename Base::storage_type, U &&>
  UnsafeRef(U &&b) : Base(std::forward<U>(b)) {}

  // ── Pure pass-through coefficient access ─────────────────────────

  template <typename... Args>
  auto coeff(Args &&...args) const -> value_type {
    return expression().coeff(std::forward<Args>(args)...);
  }

  template <typename... Args>
  auto coeff_ref(Args &&...args) -> value_type &
    requires(traits::is_assignable())
  {
    return expression().coeff_ref(std::forward<Args>(args)...);
  }

  template <typename... Args>
  auto const_coeff_ref(Args &&...args) const -> const value_type &
    requires(traits::is_referrable())
  {
    return expression().const_coeff_ref(std::forward<Args>(args)...);
  }

  // ── Assignment forwarding ────────────────────────────────────────

  template <zipper::concepts::Expression V>
  void assign(const V &v)
    requires(
        traits::is_assignable() &&
        extents_traits::template is_convertable_from<
            typename zipper::expression::detail::ExpressionTraits<
                V>::extents_type>())
  {
    expression().assign(v);
  }

  // ── Owned copy ──────────────────────────────────────────────────

  /// make_owned() just delegates to the child — UnsafeRef is transparent.
  auto make_owned() const {
    return expression().make_owned();
  }

  // ── Nonzero range forwarding ──────────────────────────────────────

  /// Forward nonzero_range queries to the child when it has known zeros.
  template <rank_type D, typename... Args>
    requires(traits::has_known_zeros)
  auto nonzero_range(Args &&...args) const {
    return expression().template nonzero_range<D>(
        std::forward<Args>(args)...);
  }

  /// Forward col_range_for_row to child (rank-2 convenience).
  auto col_range_for_row(index_type row) const
    requires(traits::has_known_zeros && extents_type::rank() == 2)
  {
    return expression().col_range_for_row(row);
  }

  /// Forward row_range_for_col to child (rank-2 convenience).
  auto row_range_for_col(index_type col) const
    requires(traits::has_known_zeros && extents_type::rank() == 2)
  {
    return expression().row_range_for_col(col);
  }

  /// Forward nonzero_segment to child (rank-1 convenience).
  auto nonzero_segment() const
    requires(traits::has_known_zeros && extents_type::rank() == 1)
  {
    return expression().nonzero_segment();
  }
};

// Deduction guides
template <zipper::concepts::Expression ExprType>
UnsafeRef(const ExprType &) -> UnsafeRef<const ExprType &>;

template <zipper::concepts::Expression ExprType>
UnsafeRef(ExprType &) -> UnsafeRef<ExprType &>;

} // namespace unary
} // namespace zipper::expression
#endif
