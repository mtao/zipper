#if !defined(ZIPPER_EXPRESSION_BINARY_EXTERIORPRODUCT_HPP)
#define ZIPPER_EXPRESSION_BINARY_EXTERIORPRODUCT_HPP

#include "BinaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/assert.hpp"
#include "zipper/exterior/detail/combinatorics.hpp"

#include <algorithm>
#include <array>
#include <bit>

namespace zipper::expression {
namespace binary {

template <zipper::concepts::QualifiedExpression A,
          zipper::concepts::QualifiedExpression B>
class ExteriorProduct;

namespace _detail_exterior {

template <typename A, typename B>
struct exterior_coeffwise_extents_values;

template <index_type... A, index_type... B>
struct exterior_coeffwise_extents_values<extents<A...>, extents<B...>> {
    using product_extents_type = zipper::extents<A..., B...>;
    using a_extents_type = extents<A...>;
    using b_extents_type = extents<B...>;
};

} // namespace _detail_exterior
} // namespace binary

template <zipper::concepts::QualifiedExpression A,
          zipper::concepts::QualifiedExpression B>
struct detail::ExpressionDetail<binary::ExteriorProduct<A, B>>
    : public binary::detail::DefaultBinaryExpressionDetail<A, B> {
    using _Base = binary::detail::DefaultBinaryExpressionDetail<A, B>;
    using ATraits = typename _Base::ATraits;
    using BTraits = typename _Base::BTraits;

    constexpr static rank_type lhs_rank = ATraits::extents_type::rank();
    constexpr static rank_type rhs_rank = BTraits::extents_type::rank();
    constexpr static rank_type rank = lhs_rank + rhs_rank;
    using CEV = binary::_detail_exterior::exterior_coeffwise_extents_values<
        typename ATraits::extents_type, typename BTraits::extents_type>;
};

template <zipper::concepts::QualifiedExpression A,
          zipper::concepts::QualifiedExpression B>
struct detail::ExpressionTraits<binary::ExteriorProduct<A, B>>
    : public binary::detail::DefaultBinaryExpressionTraits<A, B> {
    using _Detail = detail::ExpressionDetail<binary::ExteriorProduct<A, B>>;
    using extents_type = typename _Detail::CEV::product_extents_type;
    constexpr static bool is_value_based = false;
    constexpr static bool is_coefficient_consistent = false;
};

namespace binary {

template <zipper::concepts::QualifiedExpression A,
          zipper::concepts::QualifiedExpression B>
class ExteriorProduct : public BinaryExpressionBase<ExteriorProduct<A, B>, A, B> {
  public:
    using self_type = ExteriorProduct<A, B>;
    using Base = BinaryExpressionBase<self_type, A, B>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using detail_type = zipper::expression::detail::ExpressionDetail<self_type>;
    using value_type = typename traits::value_type;
    using extents_type = typename traits::extents_type;
    using Base::lhs;
    using Base::rhs;

    constexpr static rank_type lhs_rank = detail_type::lhs_rank;
    constexpr static rank_type rhs_rank = detail_type::rhs_rank;
    constexpr static rank_type rank = detail_type::rank;

    template <typename U, typename V>
        requires std::constructible_from<typename Base::lhs_storage_type, U &&> &&
                 std::constructible_from<typename Base::rhs_storage_type, V &&>
    ExteriorProduct(U &&a, V &&b) : Base(std::forward<U>(a), std::forward<V>(b)) {
        ZIPPER_ASSERT(lhs_rank == 0 || rhs_rank == 0 || lhs().extent(0) == rhs().extent(0));
    }

    constexpr auto extent(rank_type i) const -> index_type {
        if (i < lhs_rank) {
            return lhs().extent(i);
        }
        return rhs().extent(i - lhs_rank);
    }

    constexpr auto extents() const -> extents_type {
        return zipper::detail::ExtentsTraits<extents_type>::make_extents_from(*this);
    }

    template <typename... Args>
    auto coeff(Args &&...args) const -> value_type {
        static_assert(sizeof...(Args) == rank,
                      "ExteriorProduct::coeff requires full index pack");
        std::array<index_type, rank> indices{static_cast<index_type>(args)...};

        if (exterior::detail::has_duplicate_indices(indices)) {
            return value_type{0};
        }

        value_type sum = value_type{0};
        auto mask_limit = index_type{1} << rank;
        for (index_type mask = 0; mask < mask_limit; ++mask) {
            if (std::popcount(mask) != lhs_rank) {
                continue;
            }

            std::array<index_type, lhs_rank == 0 ? 1 : lhs_rank> lhs_indices{};
            std::array<index_type, rhs_rank == 0 ? 1 : rhs_rank> rhs_indices{};
            index_type lhs_pos = 0;
            index_type rhs_pos = 0;
            for (index_type j = 0; j < rank; ++j) {
                if ((mask & (index_type{1} << j)) != 0) {
                    if constexpr (lhs_rank > 0) {
                        lhs_indices[lhs_pos++] = indices[j];
                    }
                } else {
                    if constexpr (rhs_rank > 0) {
                        rhs_indices[rhs_pos++] = indices[j];
                    }
                }
            }

            const int sign = shuffle_sign(mask);
            sum += value_type(sign) * evaluate_lhs(lhs_indices) * evaluate_rhs(rhs_indices);
        }

        return sum;
    }

    auto make_owned() const {
        auto owned_a = lhs().make_owned();
        auto owned_b = rhs().make_owned();
        return ExteriorProduct<decltype(owned_a), decltype(owned_b)>(
            std::move(owned_a), std::move(owned_b));
    }

  private:
    static auto shuffle_sign(index_type mask) -> int {
        int inversions = 0;
        index_type seen_rhs = 0;
        for (index_type j = 0; j < rank; ++j) {
            if ((mask & (index_type{1} << j)) == 0) {
                ++seen_rhs;
            } else {
                inversions += static_cast<int>(seen_rhs);
            }
        }
        return (inversions % 2 == 0) ? 1 : -1;
    }

    template <typename Array, std::size_t... N>
    auto evaluate_lhs_impl(const Array &indices, std::index_sequence<N...>) const
        -> value_type {
        if constexpr (lhs_rank == 0) {
            return lhs()();
        } else {
            return lhs()(indices[N]...);
        }
    }

    template <typename Array>
    auto evaluate_lhs(const Array &indices) const -> value_type {
        return evaluate_lhs_impl(indices, std::make_index_sequence<lhs_rank>{});
    }

    template <typename Array, std::size_t... N>
    auto evaluate_rhs_impl(const Array &indices, std::index_sequence<N...>) const
        -> value_type {
        if constexpr (rhs_rank == 0) {
            return rhs()();
        } else {
            return rhs()(indices[N]...);
        }
    }

    template <typename Array>
    auto evaluate_rhs(const Array &indices) const -> value_type {
        return evaluate_rhs_impl(indices, std::make_index_sequence<rhs_rank>{});
    }
};

template <zipper::concepts::Expression A, zipper::concepts::Expression B>
ExteriorProduct(const A &, const B &) -> ExteriorProduct<const A &, const B &>;

} // namespace binary
} // namespace zipper::expression

#endif
