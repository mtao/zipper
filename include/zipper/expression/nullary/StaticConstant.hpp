#if !defined(ZIPPER_EXPRESSION_NULLARY_STATIC_CONSTANT_HPP)
#define ZIPPER_EXPRESSION_NULLARY_STATIC_CONSTANT_HPP

/// @file StaticConstant.hpp
/// @brief Nullary expression with a compile-time constant integer value.
/// @ingroup expressions_nullary
///
/// `StaticConstant<T, Value, Indices...>` is a rank-N expression that returns
/// `static_cast<T>(Value)` for every coefficient.  The constant is a template
/// parameter (constexpr int) because floating-point types cannot be non-type
/// template parameters.  The cast to `T` happens at runtime.
///
/// This expression owns no data — its value is encoded in the type.
///
/// Two convenience aliases are provided:
///   - `Zero<T, Indices...>`  = `StaticConstant<T, 0, Indices...>`
///   - `Ones<T, Indices...>`  = `StaticConstant<T, 1, Indices...>`
///
/// @code
///   // Static 3-vector of zeros
///   auto z = Zero<double, 3>();
///   // z(0) == 0.0, z(1) == 0.0, z(2) == 0.0
///
///   // Static 3x3 matrix of ones
///   auto o = Ones<double, 3, 3>();
///   // o(i, j) == 1.0 for all i, j
///
///   // Static 4-vector of twos
///   auto c = StaticConstant<float, 2, 4>();
///   // c(i) == 2.0f for all i
///
///   // Dynamic extents
///   auto z_dyn = Zero<double, dynamic_extent>(n);
/// @endcode
///
/// **Zero-aware sparsity:** When `Value == 0`, the expression has
/// `has_index_set = true` and returns empty index sets, enabling zero-aware
/// optimizations to skip all coefficients.

#include "zipper/expression/ExpressionBase.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/expression/detail/IndexSet.hpp"

namespace zipper::expression {
namespace nullary {

template <typename T, int Value, index_type... Indices>
class StaticConstant
    : public ExpressionBase<StaticConstant<T, Value, Indices...>>,
      public zipper::extents<Indices...> {
public:
  using self_type = StaticConstant<T, Value, Indices...>;
  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using extents_type = typename traits::extents_type;
  using extents_traits = typename traits::extents_traits;
  using value_type = typename traits::value_type;

  using extents_type::extent;
  using extents_type::rank;
  auto extents() const -> const extents_type & { return *this; }

  /// The compile-time constant value.
  constexpr static int static_value = Value;

  StaticConstant(const StaticConstant &) = default;
  StaticConstant(StaticConstant &&) = default;
  auto operator=(const StaticConstant &) -> StaticConstant & = default;
  auto operator=(StaticConstant &&) -> StaticConstant & = default;

  StaticConstant()
    requires(extents_traits::is_static)
  = default;

  StaticConstant(const extents_type &e) : extents_type(e) {}

  template <zipper::concepts::Index... Args>
  StaticConstant(Args &&...args)
      : StaticConstant(extents_type(std::forward<Args>(args)...)) {}

  /// Returns the constant value, cast to value_type.
  auto coeff(zipper::concepts::Index auto &&...) const -> value_type {
    return static_cast<value_type>(Value);
  }

  /// StaticConstant already owns its data — make_owned() returns a copy.
  auto make_owned() const -> StaticConstant { return *this; }

  // ── Index set queries (Zero only) ─────────────────────────────
  // When Value == 0, every coefficient is zero, so the index set along
  // any dimension is empty.

  template <rank_type D>
    requires(D < extents_type::rank() && Value == 0)
  auto index_set(index_type /*other_idx*/ = 0) const
      -> zipper::expression::detail::EmptyIndexRange {
    return {};
  }

  template <rank_type D>
    requires(D < extents_type::rank() && Value == 0)
  auto nonzero_range(index_type other_idx = 0) const
      -> zipper::expression::detail::EmptyIndexRange {
    return index_set<D>(other_idx);
  }

  auto col_range_for_row(index_type /*row*/) const
      -> zipper::expression::detail::EmptyIndexRange
    requires(extents_type::rank() == 2 && Value == 0)
  {
    return {};
  }

  auto row_range_for_col(index_type /*col*/) const
      -> zipper::expression::detail::EmptyIndexRange
    requires(extents_type::rank() == 2 && Value == 0)
  {
    return {};
  }
};

/// Convenience alias: zero expression.
template <typename T, index_type... Indices>
using Zero = StaticConstant<T, 0, Indices...>;

/// Convenience alias: all-ones expression.
template <typename T, index_type... Indices>
using Ones = StaticConstant<T, 1, Indices...>;

} // namespace nullary

// ── ExpressionTraits specialization ─────────────────────────────

template <typename T, int Value, index_type... Indices>
struct detail::ExpressionTraits<nullary::StaticConstant<T, Value, Indices...>>
    : public BasicExpressionTraits<
          T, zipper::extents<Indices...>,
          expression::detail::AccessFeatures{.is_const = true,
                                             .is_reference = false},
          expression::detail::ShapeFeatures{.is_resizable = true}> {

  /// Zero expressions have structurally known zero regions (everything is
  /// zero).
  constexpr static bool has_index_set = (Value == 0);

  /// Backward-compatible alias.
  constexpr static bool has_known_zeros = has_index_set;
};

} // namespace zipper::expression

#endif
