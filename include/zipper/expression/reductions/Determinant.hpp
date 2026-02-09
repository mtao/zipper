#if !defined(ZIPPER_EXPRESSION_REDUCTIONS_DETERMINANT_HPP)
#define ZIPPER_EXPRESSION_REDUCTIONS_DETERMINANT_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>
#include <type_traits>
#include <vector>

#include "detail/swap_parity.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/concepts/Index.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"
#include "zipper/utils/extents/convert_extents.hpp"

namespace zipper::expression {
namespace reductions {
namespace detail {

template <typename T>
T det2(const T &a, const T &b, const T &c, const T &d) {
  return a * d - b * c;
}
} // namespace detail

template <zipper::concepts::Expression Expr>
  requires(Expr::extents_type::rank() == 2)
class Determinant {
public:
  using self_type = Determinant<Expr>;
  using expression_type = Expr;
  using expression_traits =
      zipper::expression::detail::ExpressionTraits<expression_type>;
  using value_type = typename Expr::value_type;
  using base_extents_type = Expr::extents_type;
  using base_extents_traits =
      zipper::detail::ExtentsTraits<base_extents_type>;
  constexpr static index_type static_rows =
      base_extents_type::static_extent(0);
  constexpr static index_type static_cols =
      base_extents_type::static_extent(1);

  constexpr static size_t size =
      static_rows == std::dynamic_extent ? static_cols : static_rows;

  Determinant(const Expr &v) : m_expression(v) {}

  Determinant(Determinant &&v) = default;
  Determinant(const Determinant &v) = default;
  auto operator=(Determinant &&v) -> Determinant & = delete;
  auto operator=(const Determinant &v) -> Determinant & = delete;

  value_type operator()() const {
    if constexpr (base_extents_traits::is_dynamic) {
      assert(m_expression.extent(0) == m_expression.extent(1));
    } else {
      static_assert(static_rows == static_cols);
    }
    if constexpr (size == 1) {
      return m_expression(0, 0);
    } else if constexpr (size == 2) {
      return det2();
    } else if constexpr (size == 3) {
      return det3();
    } else {
      return naive_impl();
    }
  }

  value_type det2() const {
    return detail::det2(m_expression(0, 0), m_expression(0, 1),
                        m_expression(1, 0), m_expression(1, 1));
  }
  // borrowed from eigen determinant
  value_type det3() const {
    auto helper = [](const auto &v, index_type a, index_type b,
                     index_type c) {
      return v(0, a) * detail::det2(v(1, b), v(1, c), v(2, b), v(2, c));
    };

    return helper(m_expression, 0, 1, 2) - helper(m_expression, 1, 0, 2) +
           helper(m_expression, 2, 0, 1);
  }
  value_type naive_impl() const {
    value_type v = 0.0;
    using dat = typename std::conditional_t<size == std::dynamic_extent,
                                            std::vector<index_type>,
                                            std::array<index_type, size>>;
    auto io =
        std::ranges::views::iota(index_type(0), m_expression.extent(0));
    dat p;
    if constexpr (size == std::dynamic_extent) {
      p.resize(m_expression.extent(0));
    }
    std::ranges::copy(io.begin(), io.end(), p.begin());

    do {
      value_type x = detail::swap_parity<size>(p) ? 1 : -1;
      for (index_type j = 0; j < m_expression.extent(0); ++j) {
        const value_type &coeff = m_expression(j, p[j]);
        if (coeff == value_type(0)) {
          x = 0;
          break;
        }
        x *= coeff;
      }
      v += x;
    } while (std::ranges::next_permutation(p.begin(), p.end()).found);

    return v;
  }

private:
  const Expr &m_expression;
};

template <zipper::concepts::Expression Expr>
Determinant(const Expr &) -> Determinant<Expr>;

} // namespace reductions
} // namespace zipper::expression
#endif
