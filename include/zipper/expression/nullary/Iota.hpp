#if !defined(ZIPPER_EXPRESSION_NULLARY_IOTA_HPP)
#define ZIPPER_EXPRESSION_NULLARY_IOTA_HPP

/// @file Iota.hpp
/// @brief Nullary expression that returns the index along a chosen axis.
/// @ingroup expressions_nullary
///
/// `Iota<T, D, Indices...>` is a rank-N expression whose coefficient at
/// position `(i0, i1, ..., iN)` is `static_cast<T>(i_D)` -- i.e. the
/// D-th index, cast to the value type T.
///
/// For rank-1, this is a simple counting sequence: 0, 1, 2, ...
/// For rank-2 with D=0, every column is [0,1,...,M-1]; with D=1, every
/// row is [0,1,...,N-1].
///
/// This expression owns no data -- it generates coefficients on the fly
/// from the index arguments.
///
/// @code
///   // Static 5-vector: [0, 1, 2, 3, 4]
///   auto v = Iota<double, 0, 5>();
///   // v(3) == 3.0
///
///   // Static 3x4 matrix returning row index
///   auto rows = Iota<double, 0, 3, 4>();
///   // rows(2, 1) == 2.0
///
///   // Static 3x4 matrix returning column index
///   auto cols = Iota<double, 1, 3, 4>();
///   // cols(2, 1) == 1.0
///
///   // Dynamic extents
///   auto v_dyn = Iota<double, 0, dynamic_extent>(n);
/// @endcode
///
/// Two free functions build on Iota:
///
///   - `iota<T, D>(start, extents)` -- returns `start + Iota`, i.e. a
///     counting sequence beginning at `start` instead of 0.  Implemented
///     as `binary::Plus<Constant, Iota>`.
///
///   - `linear_space<T, D>(start, stop, extents)` -- returns evenly
///     spaced values from `start` to `stop` (inclusive) along axis D,
///     with the number of points equal to `extent(D)`.  Implemented as
///     `binary::Plus<Constant, ScalarMultiplies<Iota>>`, i.e.
///     `start + Iota * ((stop - start) / (extent(D) - 1))`.
///
/// @see Constant   -- used by `iota()` for the starting offset.
/// @see binary::Plus -- used by `iota()` and `linear_space()` to add
///                      the offset to the counting/scaled sequence.
/// @see unary::ScalarMultiplies -- used by `linear_space()` to scale
///                                 the Iota sequence by the step size.

#include "Constant.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/expression/ExpressionBase.hpp"
#include "zipper/expression/binary/ArithmeticExpressions.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/expression/unary/ScalarArithmetic.hpp"

namespace zipper::expression {
namespace nullary {

template <typename T, rank_type D = 0, index_type... Indices>
class Iota : public ExpressionBase<Iota<T, D, Indices...>>,
             public zipper::extents<Indices...> {

  // When the expression has a definite rank (rank > 0), D must be
  // within bounds.  Rank-0 (infinite) expressions accept any D since
  // the user supplies arbitrary index packs at call time.
  static_assert(sizeof...(Indices) == 0 || D < sizeof...(Indices),
                "Iota axis D must be less than the number of dimensions");

public:
  using self_type = Iota<T, D, Indices...>;
  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using extents_type = typename traits::extents_type;
  using extents_traits = typename traits::extents_traits;
  using value_type = typename traits::value_type;

  using extents_type::extent;
  using extents_type::rank;
  auto extents() const -> const extents_type & { return *this; }

  Iota(const Iota &) = default;
  Iota(Iota &&) = default;
  auto operator=(const Iota &) -> Iota & = default;
  auto operator=(Iota &&) -> Iota & = default;

  Iota()
    requires(extents_traits::is_static)
  = default;

  Iota(const extents_type &e) : extents_type(e) {}

  template <zipper::concepts::Index... Args>
  Iota(Args &&...args)
      : Iota(extents_type(std::forward<Args>(args)...)) {}

  /// Returns the D-th index, cast to value_type.
  auto coeff(zipper::concepts::Index auto &&...idxs) const -> value_type {
    return static_cast<value_type>(
        zipper::detail::pack_index<D>(idxs...));
  }

  /// Iota already owns its data -- make_owned() returns a copy.
  auto make_owned() const -> Iota { return *this; }
};

// ── Factory functions ───────────────────────────────────────────────

/// Creates a counting sequence starting at @p start along axis @p D.
///
/// Returns `Constant(start) + Iota<T, D>` as an explicit
/// `binary::Plus` expression node.
///
/// @tparam T      Value type
/// @tparam D      Axis along which to count (default 0)
/// @tparam Indices  Static extent pack
/// @param start   The first value of the sequence
/// @param e       Shape of the expression
///
/// @code
///   auto v = iota<double>(3.0, zipper::extents<5>{});
///   // v(0)==3, v(1)==4, v(2)==5, v(3)==6, v(4)==7
/// @endcode
template <typename T, rank_type D = 0, index_type... Indices>
auto iota(const T &start, const zipper::extents<Indices...> &e = {}) {
  using IotaType = Iota<T, D, Indices...>;
  using ConstType = Constant<T, Indices...>;
  using PlusType = binary::Plus<ConstType, IotaType>;
  return PlusType(ConstType(start, e), IotaType(e));
}

/// Creates evenly spaced values from @p start to @p stop (inclusive)
/// along axis @p D.
///
/// The number of points equals `e.extent(D)`.  Implemented as
/// `Constant(start) + Iota * step` where
/// `step = (stop - start) / (extent(D) - 1)`.
///
/// @tparam T      Value type
/// @tparam D      Axis along which to space values (default 0)
/// @tparam Indices  Static extent pack
/// @param start   First value
/// @param stop    Last value (inclusive)
/// @param e       Shape of the expression
///
/// @code
///   auto v = linear_space<double>(0.0, 1.0, zipper::extents<5>{});
///   // v(0)==0.0, v(1)==0.25, v(2)==0.5, v(3)==0.75, v(4)==1.0
/// @endcode
template <typename T, rank_type D = 0, index_type... Indices>
auto linear_space(const T &start, const T &stop,
                  const zipper::extents<Indices...> &e) {
  using IotaType = Iota<T, D, Indices...>;
  auto iota_expr = IotaType(e);

  T step =
      (stop - start) / static_cast<T>(iota_expr.extent(D) - 1);

  // Iota * step  (scalar on right)
  using ScaledType =
      unary::ScalarMultiplies<T, IotaType, true>;
  auto scaled = ScaledType(std::move(iota_expr), step);

  // Constant(start) + scaled
  using ConstType = Constant<T, Indices...>;
  using ResultType = binary::Plus<ConstType, ScaledType>;
  return ResultType(ConstType(start, e), std::move(scaled));
}

} // namespace nullary

// ── ExpressionTraits specialization ─────────────────────────────────

template <typename T, rank_type D, index_type... Indices>
struct detail::ExpressionTraits<nullary::Iota<T, D, Indices...>>
    : public BasicExpressionTraits<
          T, zipper::extents<Indices...>,
          expression::detail::AccessFeatures::const_value(),
          expression::detail::ShapeFeatures::resizable()> {};

} // namespace zipper::expression

#endif
