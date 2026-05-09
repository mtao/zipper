#if !defined(ZIPPER_EXPRESSION_UNARY_ALTERNATINGPROJECTION_HPP)
#define ZIPPER_EXPRESSION_UNARY_ALTERNATINGPROJECTION_HPP

#include <algorithm>
#include <array>
#include <vector>

#include "UnaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/assert.hpp"
#include "zipper/exterior/detail/combinatorics.hpp"

namespace zipper::expression {
namespace unary {

template <zipper::concepts::QualifiedExpression ExpressionType>
class AlternatingProjection;

} // namespace unary

template <zipper::concepts::QualifiedExpression ExpressionType>
struct detail::ExpressionTraits<unary::AlternatingProjection<ExpressionType>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
          ExpressionType, zipper::detail::AccessFeatures::const_value()> {
    constexpr static bool is_value_based = false;
    constexpr static bool is_coefficient_consistent = false;
};

namespace unary {

template <zipper::concepts::QualifiedExpression ExpressionType>
class AlternatingProjection
    : public UnaryExpressionBase<AlternatingProjection<ExpressionType>,
                                 ExpressionType> {
  public:
    using self_type = AlternatingProjection<ExpressionType>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using value_type = typename traits::value_type;
    using extents_type = typename traits::extents_type;
    using Base = UnaryExpressionBase<self_type, ExpressionType>;
    using Base::expression;

    template <typename U>
        requires std::constructible_from<typename Base::storage_type, U &&>
    AlternatingProjection(U &&b) : Base(std::forward<U>(b)) {}

    auto make_owned() const {
        auto owned_child = expression().make_owned();
        return AlternatingProjection<const decltype(owned_child)>(
            std::move(owned_child));
    }

    template <typename... Args>
    auto coeff(Args &&...args) const -> value_type {
        constexpr rank_type rank = extents_type::rank();
        static_assert(sizeof...(Args) == rank,
                      "AlternatingProjection::coeff requires full index pack");

        std::array<index_type, rank> indices{static_cast<index_type>(args)...};
        if (exterior::detail::has_duplicate_indices(indices)) {
            return value_type{0};
        }

        std::array<index_type, rank> sorted = indices;
        std::ranges::sort(sorted);

        value_type sum = value_type{0};
        do {
            std::array<index_type, rank> permutation{};
            for (index_type j = 0; j < rank; ++j) {
                for (index_type k = 0; k < rank; ++k) {
                    if (sorted[k] == indices[j]) {
                        permutation[j] = k;
                        break;
                    }
                }
            }
            const int sign = exterior::detail::inversion_parity(permutation);
            sum += value_type(sign) * evaluate(sorted);
        } while (std::ranges::next_permutation(sorted).found);

        return sum / value_type(exterior::detail::factorial(rank));
    }

  private:
    template <std::size_t... N>
    auto evaluate(const std::array<index_type, extents_type::rank()> &indices,
                  std::index_sequence<N...>) const -> value_type {
        return expression()(indices[N]...);
    }

    auto evaluate(const std::array<index_type, extents_type::rank()> &indices)
        const -> value_type {
        return evaluate(indices,
                        std::make_index_sequence<extents_type::rank()>{});
    }
};

template <zipper::concepts::QualifiedExpression ExpressionType>
AlternatingProjection(const ExpressionType &)
    -> AlternatingProjection<const ExpressionType &>;

} // namespace unary
} // namespace zipper::expression

#endif
