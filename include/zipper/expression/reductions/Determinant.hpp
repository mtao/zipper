#if !defined(ZIPPER_EXPRESSION_REDUCTIONS_DETERMINANT_HPP)
#define ZIPPER_EXPRESSION_REDUCTIONS_DETERMINANT_HPP

#include "zipper/detail/assert.hpp"
#include <algorithm>
#include <array>
#include <ranges>
#include <type_traits>
#include <vector>

#include "ReductionBase.hpp"
#include "detail/swap_parity.hpp"
#include "zipper/concepts/Index.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/utils/extents/convert_extents.hpp"

namespace zipper::expression {
namespace reductions {
    namespace detail {

        template <typename T>
        T det2(const T &a, const T &b, const T &c, const T &d) {
            return a * d - b * c;
        }
    } // namespace detail

    template <typename Expr>
    class Determinant : public ReductionBase<Determinant<Expr>, Expr> {
      public:
        using Base = ReductionBase<Determinant<Expr>, Expr>;
        using typename Base::expression_traits;
        using typename Base::expression_type;
        using typename Base::value_type;

        using Base::Base;
        using Base::expression;

        using base_extents_type = expression_type::extents_type;
        using base_extents_traits =
            zipper::detail::ExtentsTraits<base_extents_type>;
        constexpr static index_type static_rows =
            base_extents_type::static_extent(0);
        constexpr static index_type static_cols =
            base_extents_type::static_extent(1);

        constexpr static size_t size =
            static_rows == std::dynamic_extent ? static_cols : static_rows;

        value_type operator()() const {
            if constexpr (base_extents_traits::is_dynamic) {
                ZIPPER_ASSERT(expression().extent(0) == expression().extent(1));
            } else {
                static_assert(static_rows == static_cols);
            }
            if constexpr (size == 1) {
                return expression()(0, 0);
            } else if constexpr (size == 2) {
                return det2();
            } else if constexpr (size == 3) {
                return det3();
            } else {
                return naive_impl();
            }
        }

        value_type det2() const {
            return detail::det2(expression()(0, 0),
                                expression()(0, 1),
                                expression()(1, 0),
                                expression()(1, 1));
        }
        // borrowed from eigen determinant
        value_type det3() const {
            auto helper =
                [](const auto &v, index_type a, index_type b, index_type c) {
                    return v(0, a)
                           * detail::det2(v(1, b), v(1, c), v(2, b), v(2, c));
                };

            return helper(expression(), 0, 1, 2) - helper(expression(), 1, 0, 2)
                   + helper(expression(), 2, 0, 1);
        }
        value_type naive_impl() const {
            value_type v = 0.0;
            using dat =
                typename std::conditional_t<size == std::dynamic_extent,
                                            std::vector<index_type>,
                                            std::array<index_type, size>>;
            auto io =
                std::ranges::views::iota(index_type(0), expression().extent(0));
            dat p;
            if constexpr (size == std::dynamic_extent) {
                p.resize(expression().extent(0));
            }
            std::ranges::copy(io.begin(), io.end(), p.begin());

            do {
                value_type x = detail::swap_parity<size>(p) ? 1 : -1;
                for (index_type j = 0; j < expression().extent(0); ++j) {
                    const value_type &coeff = expression()(j, p[j]);
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
    };

    template <zipper::concepts::Expression E>
    Determinant(E &) -> Determinant<E &>;

    template <zipper::concepts::Expression E>
    Determinant(E &&) -> Determinant<E>;

} // namespace reductions
} // namespace zipper::expression
#endif
