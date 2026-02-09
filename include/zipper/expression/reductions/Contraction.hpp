#if !defined(ZIPPER_EXPRESSION_REDUCTIONS_CONTRACTION_HPP)
#define ZIPPER_EXPRESSION_REDUCTIONS_CONTRACTION_HPP

#include "zipper/concepts/Expression.hpp"
#include "zipper/concepts/Index.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"
#include "zipper/utils/extents/convert_extents.hpp"

namespace zipper::expression {
namespace reductions {

namespace detail {
template <typename>
struct folded_extents;

template <index_type... N>
struct folded_extents<extents<N...>> {
  using base_extents_type = extents<N...>;
  constexpr static std::size_t base_rank = sizeof...(N);

  constexpr static std::size_t rank = base_rank / 2;

  template <typename>
  struct half_;
  template <index_type... M>
  struct half_<std::integer_sequence<rank_type, M...>> {
    using extents_type =
        extents<(base_extents_type::static_extent(M) == std::dynamic_extent
                     ? base_extents_type::static_extent(base_rank - M)
                     : base_extents_type::static_extent(M))...>;

    extents_type get_extents(const base_extents_type &e) {
      return zipper::utils::extents::convert_extents<base_extents_type, M...>(
          e);
    }
  };

  using half =
      half_<decltype(std::make_integer_sequence<rank_type, base_rank / 2>{})>;
  using extents_type = half::extents_type;

  template <rank_type M, concepts::IndexPackTuple tuple_type>
  auto get_arg(const tuple_type &t) {
    constexpr static rank_type I = M >= rank ? base_rank - M - 1 : M;
    return std::get<I>(t);
  }
};

} // namespace detail

template <zipper::concepts::Expression Expr>
class Contraction {
public:
  using self_type = Contraction<Expr>;
  using expression_type = Expr;
  using expression_traits =
      zipper::expression::detail::ExpressionTraits<expression_type>;
  using value_type = typename Expr::value_type;
  using base_extents_type = Expr::extents_type;
  using folder = detail::folded_extents<base_extents_type>;
  using extents_type = folder::extents_type;

  Contraction(const Expr &v) : m_expression(v) {}

  Contraction(Contraction &&v) = default;
  Contraction(const Contraction &v) = default;
  auto operator=(Contraction &&v) -> Contraction & = delete;
  auto operator=(const Contraction &v) -> Contraction & = delete;

  template <rank_type... N>
  value_type coeff_(concepts::IndexPackTuple auto const &t,
                    std::integer_sequence<rank_type, N...>) {
    return m_expression(folder::template get_arg<N>(t)...);
  }

  value_type operator()() const {
    value_type v = 0.0;
    for (const auto &i :
         zipper::utils::extents::all_extents_indices(m_extents)) {
      v += coeff_(
          i,
          std::make_integer_sequence<rank_type, extents_type::rank()>{});
    }
    return v;
  }

private:
  const Expr &m_expression;
  extents_type m_extents;
};

template <zipper::concepts::Expression Expr>
Contraction(const Expr &) -> Contraction<Expr>;

} // namespace reductions
} // namespace zipper::expression
#endif
