#if !defined(ZIPPER_EXPRESSION_UNARY_PARTIALTRACE_HPP)
#define ZIPPER_EXPRESSION_UNARY_PARTIALTRACE_HPP

#include "DiagonalExtract.hpp"
#include "Slice.hpp"
#include "UnaryExpressionBase.hpp"
#include "detail/invert_integer_sequence.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/extents/static_extents_to_array.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/expression/reductions/CoefficientSum.hpp"

namespace zipper::expression {
namespace unary {
template <zipper::concepts::QualifiedExpression ExprType, rank_type... Indices>
    requires(sizeof...(Indices) % 2 == 0)
class PartialTrace;

}

/// Implementation details for PartialTrace expressions.
///
/// Holds the index remover utility that maps between the traced and full
/// index spaces. The class body needs this for the constructor (to
/// compute output extents), slice_type computation, and coeff dispatching.
template <zipper::concepts::QualifiedExpression ExprType, rank_type... Indices>
struct detail::ExpressionDetail<unary::PartialTrace<ExprType, Indices...>> {
    using Base = detail::ExpressionTraits<std::decay_t<ExprType>>;
    using index_remover =
        unary::detail::invert_integer_sequence<Base::extents_type::rank(),
                                               Indices...>;
};

template <zipper::concepts::QualifiedExpression ExprType, rank_type... Indices>
struct detail::ExpressionTraits<unary::PartialTrace<ExprType, Indices...>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
          ExprType,
          zipper::detail::AccessFeatures::const_value()> {
    using _Detail = detail::ExpressionDetail<unary::PartialTrace<ExprType, Indices...>>;
    using extents_type = typename _Detail::index_remover::template assign_types<
        zipper::extents, zipper::detail::extents::static_extents_to_array_v<
                             typename std::decay_t<ExprType>::extents_type>>;
    using value_type = typename _Detail::Base::value_type;
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
};

namespace unary {
// indices are the indices being traced
template <zipper::concepts::QualifiedExpression ExprType, rank_type... Indices>
    requires(sizeof...(Indices) % 2 == 0)
class PartialTrace
    : public UnaryExpressionBase<PartialTrace<ExprType, Indices...>, ExprType> {
   public:
    using self_type = PartialTrace<ExprType, Indices...>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using detail_type = zipper::expression::detail::ExpressionDetail<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;
    using Base = UnaryExpressionBase<self_type, ExprType>;
    using Base::expression;

    template <typename U>
      requires std::constructible_from<typename Base::storage_type, U &&>
    PartialTrace(U &&b)
        : Base(std::forward<U>(b)), m_extents(detail_type::index_remover::get_extents(expression().extents())) {}
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
            Slice<const std::decay_t<ExprType>&,
                      std::conditional_t<detail_type::index_remover::in_sequence(N),
                                         rank_type, full_extent_t>...>;
    };

    using slice_type = slice_type_<std::decay_t<
        decltype(std::make_integer_sequence<
                 rank_type, std::decay_t<ExprType>::extents_type::rank()>{})>>::type;

    template <typename... Args, rank_type N>
    auto get_index(std::integral_constant<rank_type, N>, Args&&... idxs) const {
        if constexpr (((N == Indices) || ...)) {
            return zipper::full_extent_t{};
        } else {
            constexpr static rank_type Index =
                detail_type::index_remover::full_rank_to_reduced_indices[N];
            static_assert(Index <= sizeof...(Args));
            return zipper::detail::pack_index<Index>(
                std::forward<Args>(idxs)...);
        }
    }

    template <typename... Args, rank_type... N>
    value_type _coeff(std::integer_sequence<rank_type, N...>,
                      Args&&... idxs) const
        requires(sizeof...(N) == std::decay_t<ExprType>::extents_type::rank())
    {
        const auto slice =
            slice_type(expression(), get_index(std::integral_constant<rank_type, N>{},
                                         std::forward<Args>(idxs)...)...);

        DiagonalExtract<const slice_type&> diag(slice);
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
                                           std::decay_t<ExprType>::extents_type::rank()>{},
                std::forward<Args>(idxs)...);
        }
    }

private:
    extents_type m_extents;

public:
    /// Recursively deep-copy child so the result owns all data.
    auto make_owned() const {
        auto owned_child = expression().make_owned();
        return PartialTrace<const decltype(owned_child), Indices...>(
            std::move(owned_child));
    }
};

}  // namespace unary
}  // namespace zipper::expression
#endif
