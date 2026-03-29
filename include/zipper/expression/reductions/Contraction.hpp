#if !defined(ZIPPER_EXPRESSION_REDUCTIONS_CONTRACTION_HPP)
#define ZIPPER_EXPRESSION_REDUCTIONS_CONTRACTION_HPP

#include "ReductionBase.hpp"
#include "zipper/concepts/Index.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"

namespace zipper::expression {
namespace reductions {

namespace detail {

/// Given an even-rank extents type, compute the "folded" extents — the
/// first half of the index dimensions.  A full contraction iterates over
/// these dimensions and mirrors the indices: for a rank-2R tensor, index
/// tuple (i0,...,i_{R-1}) maps to `expr(i0,...,i_{R-1}, i_{R-1},...,i0)`.
template <typename>
struct folded_extents;

template <index_type... N>
struct folded_extents<extents<N...>> {
  using base_extents_type = extents<N...>;
  constexpr static std::size_t base_rank = sizeof...(N);
  constexpr static std::size_t rank = base_rank / 2;

  static_assert(base_rank % 2 == 0,
                "full contraction requires even rank");

  template <typename>
  struct half_;
  template <rank_type... M>
  struct half_<std::integer_sequence<rank_type, M...>> {
    // The folded extents take the first `rank` static extents.
    // If a static extent is dynamic, try to use the mirrored extent instead.
    using extents_type =
        extents<(base_extents_type::static_extent(M) == std::dynamic_extent
                     ? base_extents_type::static_extent(base_rank - 1 - M)
                     : base_extents_type::static_extent(M))...>;

    /// Extract the first half of the runtime extents from the base.
    static extents_type get_extents(const base_extents_type &e) {
      if constexpr (extents_type::rank_dynamic() == 0) {
        return extents_type{};
      } else {
        std::array<index_type, extents_type::rank_dynamic()> dyn;
        rank_type idx = 0;
        ((extents_type::static_extent(M) == std::dynamic_extent
              ? void(dyn[idx++] = e.extent(M))
              : void()),
         ...);
        return extents_type(dyn);
      }
    }
  };

  using half =
      half_<decltype(std::make_integer_sequence<rank_type, rank>{})>;
  using extents_type = typename half::extents_type;

  /// Map iteration index M to the corresponding full-tensor index.
  /// For M < rank, returns M (first half).
  /// For M >= rank, returns the mirror: base_rank - 1 - M.
  template <rank_type M, zipper::concepts::IndexPackTuple tuple_type>
  static auto get_arg(const tuple_type &t) {
    constexpr static rank_type I = M >= rank ? base_rank - 1 - M : M;
    return std::get<I>(t);
  }
};

} // namespace detail

/// Full contraction (fold-in-half trace) of an even-rank expression.
///
/// For a rank-2R expression, sums `expr(i0,...,i_{R-1}, i_{R-1},...,i0)`
/// over all valid index tuples.  This generalizes the matrix trace to
/// arbitrary even-rank tensors.
///
/// The result is a scalar (value_type).
template <typename Expr>
class Contraction : public ReductionBase<Contraction<Expr>, Expr> {
public:
  using Base = ReductionBase<Contraction<Expr>, Expr>;
  using typename Base::expression_type;
  using typename Base::expression_traits;
  using typename Base::value_type;

  using Base::expression;

  using base_extents_type = typename expression_type::extents_type;
  using folder = detail::folded_extents<base_extents_type>;
  using extents_type = typename folder::extents_type;

  template <typename U>
    requires std::constructible_from<typename Base::storage_type, U &&>
  Contraction(U &&v)
      : Base(std::forward<U>(v)),
        m_extents(folder::half::get_extents(expression().extents())) {}

  Contraction(Contraction &&) = default;
  Contraction(const Contraction &) = default;
  auto operator=(Contraction &&) -> Contraction & = delete;
  auto operator=(const Contraction &) -> Contraction & = delete;

  value_type operator()() const {
    value_type v = 0;
    for (const auto &i :
         zipper::utils::extents::all_extents_indices(m_extents)) {
      v += coeff_(
          i,
          std::make_integer_sequence<rank_type, base_extents_type::rank()>{});
    }
    return v;
  }

private:
  template <rank_type... N>
  value_type coeff_(zipper::concepts::IndexPackTuple auto const &t,
                    std::integer_sequence<rank_type, N...>) const {
    return expression()(folder::template get_arg<N>(t)...);
  }

  extents_type m_extents;
};

template <zipper::concepts::QualifiedExpression E>
Contraction(E &) -> Contraction<E &>;

template <zipper::concepts::QualifiedExpression E>
Contraction(E &&) -> Contraction<E>;

} // namespace reductions
} // namespace zipper::expression
#endif
