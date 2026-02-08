#if !defined(ZIPPER_EXPRESSION_UNARY_PARTIALTRACE_HPP)
#define ZIPPER_EXPRESSION_UNARY_PARTIALTRACE_HPP

#include "Diagonal.hpp"
#include "Slice.hpp"
#include "UnaryExpressionBase.hpp"
#include "detail/invert_integer_sequence.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/utils/extents/extents_formatter.hpp"
#include "zipper/detail/extents/static_extents_to_array.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/expression/reductions/CoefficientSum.hpp"

namespace zipper::expression {
namespace unary {
template <zipper::concepts::Expression ExprType, rank_type... Indices>
    requires(sizeof...(Indices) % 2 == 0)
class PartialTrace;

}
template <zipper::concepts::Expression ExprType, rank_type... Indices>
struct detail::ExpressionTraits<unary::PartialTrace<ExprType, Indices...>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<ExprType> {
    using Base = detail::ExpressionTraits<ExprType>;
    using index_remover =
        unary::detail::invert_integer_sequence<Base::extents_type::rank(),
                                               Indices...>;
    using extents_type = typename index_remover::template assign_types<
        zipper::extents, zipper::detail::extents::static_extents_to_array_v<
                             typename ExprType::extents_type>>;
    using summed_extents_type = zipper::extents<Indices...>;
    using value_type = Base::value_type;
    constexpr static bool is_writable = false;
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
    // PartialTrace computes aggregate values — not referrable or assignable
    constexpr static zipper::detail::AccessFeatures access_features = {
        .is_const = true,
        .is_reference = false,
        .is_alias_free = Base::access_features.is_alias_free,
    };
    consteval static auto is_const_valued() -> bool {
      return access_features.is_const;
    }
    consteval static auto is_reference_valued() -> bool {
      return access_features.is_reference;
    }
    consteval static auto is_assignable() -> bool {
      return !is_const_valued() && is_reference_valued();
    }
    consteval static auto is_referrable() -> bool {
      return access_features.is_reference;
    }
};

namespace unary {
// indices are the indices being traced
template <zipper::concepts::Expression ExprType, rank_type... Indices>
    requires(sizeof...(Indices) % 2 == 0)
class PartialTrace
    : public UnaryExpressionBase<PartialTrace<ExprType, Indices...>, ExprType> {
   public:
    using self_type = PartialTrace<ExprType, Indices...>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;
    using Base = UnaryExpressionBase<self_type, ExprType>;
    using Base::expression;

    PartialTrace(const ExprType& b)
        : Base(b), m_extents(traits::index_remover::get_extents(b.extents())) {}
    PartialTrace() = delete;
    PartialTrace& operator=(const PartialTrace&) = delete;
    PartialTrace& operator=(PartialTrace&&) = delete;
    PartialTrace(PartialTrace&& o) = default;
    PartialTrace(const PartialTrace& o) = default;

    constexpr auto extent(rank_type i) const -> index_type {
        return m_extents.extent(i);
    }

    constexpr auto extents() const -> extents_type {
        return m_extents;
    }

    template <typename>
    struct slice_type_;
    template <rank_type... N>
    struct slice_type_<std::integer_sequence<rank_type, N...>> {
        using type =
            Slice<const ExprType,
                      std::conditional_t<traits::index_remover::in_sequence(N),
                                         rank_type, full_extent_t>...>;
    };

    using slice_type = slice_type_<std::decay_t<
        decltype(std::make_integer_sequence<
                 rank_type, ExprType::extents_type::rank()>{})>>::type;

    template <typename... Args, rank_type N>
    auto get_index(std::integral_constant<rank_type, N>, Args&&... idxs) const {
        if constexpr (((N == Indices) || ...)) {
            return zipper::full_extent_t{};
        } else {
            constexpr static rank_type Index =
                traits::index_remover::full_rank_to_reduced_indices[N];
            static_assert(Index <= sizeof...(Args));
            return zipper::detail::pack_index<Index>(
                std::forward<Args>(idxs)...);
        }
    }

    template <typename... Args, rank_type... N>
    value_type _coeff(std::integer_sequence<rank_type, N...>,
                      Args&&... idxs) const
        requires(sizeof...(N) == ExprType::extents_type::rank())
    {
        const auto slice =
            slice_type(expression(), get_index(std::integral_constant<rank_type, N>{},
                                         std::forward<Args>(idxs)...)...);

        Diagonal<const slice_type> diag(slice);
        return reductions::CoefficientSum(diag)();
    }

    template <typename... Args>
    value_type coeff(Args&&... idxs) const {
        zipper::utils::extents::indices_in_range(this->extents(), idxs...);
        if constexpr (sizeof...(Indices) == 0) {
            return expression().coeff(std::forward<Args>(idxs)...);
        } else {
            return _coeff(
                std::make_integer_sequence<rank_type,
                                           ExprType::extents_type::rank()>{},
                std::forward<Args>(idxs)...);
        }
    }

private:
    extents_type m_extents;
};

}  // namespace unary
}  // namespace zipper::expression
#endif
