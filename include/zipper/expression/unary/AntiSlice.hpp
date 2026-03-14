#if !defined(ZIPPER_EXPRESSION_UNARY_ANTISLICE_HPP)
#define ZIPPER_EXPRESSION_UNARY_ANTISLICE_HPP

#include "UnaryExpressionBase.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/expression/detail/AssignHelper.hpp"

#include <array>

namespace zipper::expression {
namespace unary {

/// AntiSlice embeds an expression into a higher-rank space by inserting
/// new dimensions of extent 1 at the specified positions.
///
/// For example, given a Vector<T,3> (rank 1) and InsertedDim = 1, the
/// result is a rank-2 expression with extents (3, 1) — conceptually a
/// column matrix.  With InsertedDim = 0 the result is (1, 3) — a row.
///
/// This is the inverse of slicing away a dimension with a fixed index.
///
/// Template parameters:
///   ExprType      — the child expression type (possibly const-qualified,
///                   satisfying QualifiedExpression)
///   InsertedDims  — rank positions in the OUTPUT where new size-1
///                   dimensions are inserted. Must be in [0, output_rank).
template <zipper::concepts::QualifiedExpression ExprType, rank_type... InsertedDims>
class AntiSlice;

namespace _detail_antislice {

/// Compute the output rank given child rank and number of inserted dims.
template <rank_type ChildRank, rank_type NInserted>
constexpr rank_type output_rank_v = ChildRank + NInserted;

/// Build a compile-time map from output rank index → child rank index.
/// Inserted positions map to dynamic_extent (sentinel = "not from child").
/// Non-inserted positions map to consecutive child indices 0,1,2,...
template <rank_type OutputRank, rank_type... Inserted>
consteval auto make_child_index_map()
    -> std::array<rank_type, OutputRank> {
    std::array<rank_type, OutputRank> map{};
    // Mark inserted positions
    constexpr rank_type inserted_arr[] = {Inserted...};
    for (rank_type i = 0; i < sizeof...(Inserted); ++i) {
        map[inserted_arr[i]] = std::dynamic_extent;
    }
    // Fill in child indices for non-inserted positions
    rank_type child_idx = 0;
    for (rank_type i = 0; i < OutputRank; ++i) {
        if (map[i] != std::dynamic_extent) {
            map[i] = child_idx++;
        }
    }
    return map;
}

/// Build the output static extents array given child extents type and
/// inserted dim positions.
template <typename ChildExtents, rank_type OutputRank, rank_type... Inserted>
consteval auto make_static_extents()
    -> std::array<index_type, OutputRank> {
    constexpr auto child_map = make_child_index_map<OutputRank, Inserted...>();
    std::array<index_type, OutputRank> result{};
    for (rank_type i = 0; i < OutputRank; ++i) {
        if (child_map[i] == std::dynamic_extent) {
            result[i] = 1;  // inserted dim always has extent 1
        } else {
            result[i] = ChildExtents::static_extent(child_map[i]);
        }
    }
    return result;
}

/// Build the extents type for the output.
template <typename ChildExtents, rank_type OutputRank, rank_type... Inserted>
struct OutputExtentsHelper {
    static constexpr auto static_exts =
        make_static_extents<ChildExtents, OutputRank, Inserted...>();

    template <typename>
    struct Builder;

    template <rank_type... Rs>
    struct Builder<std::integer_sequence<rank_type, Rs...>> {
        using type = zipper::extents<static_exts[Rs]...>;
    };

    using type = typename Builder<
        std::make_integer_sequence<rank_type, OutputRank>>::type;
};

}  // namespace _detail_antislice

}  // namespace unary

/// ExpressionTraits specialization for AntiSlice.
/// Note: ExprType may be const-qualified but not reference-qualified.
template <zipper::concepts::QualifiedExpression ExprType, rank_type... InsertedDims>
  requires(((InsertedDims <
             unary::_detail_antislice::output_rank_v<
                 detail::ExpressionTraits<std::decay_t<ExprType>>::extents_type::rank(),
                 sizeof...(InsertedDims)>) && ...))
struct detail::ExpressionTraits<unary::AntiSlice<ExprType, InsertedDims...>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
          ExprType> {
    using StrippedExprType = std::decay_t<ExprType>;
    using Base = detail::ExpressionTraits<StrippedExprType>;
    using value_type = typename Base::value_type;
    using child_extents_type = typename Base::extents_type;

    static constexpr rank_type child_rank = child_extents_type::rank();
    static constexpr rank_type n_inserted = sizeof...(InsertedDims);
    static constexpr rank_type output_rank =
        unary::_detail_antislice::output_rank_v<child_rank, n_inserted>;

    using extents_type = typename unary::_detail_antislice::OutputExtentsHelper<
        child_extents_type, output_rank, InsertedDims...>::type;

    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;

    static constexpr auto child_index_map =
        unary::_detail_antislice::make_child_index_map<output_rank,
                                                       InsertedDims...>();
};

namespace unary {

template <zipper::concepts::QualifiedExpression ExprType, rank_type... InsertedDims>
class AntiSlice
    : public UnaryExpressionBase<AntiSlice<ExprType, InsertedDims...>,
                                 ExprType> {
   public:
    using self_type = AntiSlice<ExprType, InsertedDims...>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using extents_type = typename traits::extents_type;
    using value_type = typename traits::value_type;
    using StrippedExprType = std::decay_t<ExprType>;
    using Base = UnaryExpressionBase<self_type, ExprType>;
    using Base::expression;
    using child_traits = typename traits::Base;
    using child_extents_type = typename child_traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    static constexpr rank_type output_rank = traits::output_rank;
    static constexpr auto child_index_map = traits::child_index_map;

    AntiSlice(const AntiSlice &o) : AntiSlice(o.expression()) {}
    AntiSlice(AntiSlice &&o) : AntiSlice(o.expression()) {}

    auto operator=(const AntiSlice &) -> AntiSlice & = delete;
    auto operator=(AntiSlice &&) -> AntiSlice & = delete;

    template <typename U>
      requires std::constructible_from<typename Base::storage_type, U &&>
    AntiSlice(U &&b) : Base(std::forward<U>(b)) {}

    /// Recursively deep-copy child so the result owns all data.
    auto make_owned() const {
        auto owned_child = expression().make_owned();
        return AntiSlice<const decltype(owned_child), InsertedDims...>(
            std::move(owned_child));
    }

    constexpr auto extent(rank_type i) const -> index_type {
        if (child_index_map[i] == std::dynamic_extent) {
            return 1;  // inserted dimension
        }
        return expression().extent(child_index_map[i]);
    }

    constexpr auto extents() const -> extents_type {
        return extents_traits::make_extents_from(*this);
    }

    // ---- coefficient access ----

    template <typename Tuple, rank_type... ChildRanks>
    auto _coeff_dispatch(const Tuple &idxs,
                         std::integer_sequence<rank_type, ChildRanks...>) const
        -> value_type {
        return expression().coeff(
            _get_child_index<ChildRanks>(idxs)...);
    }

    template <typename Tuple, rank_type... ChildRanks>
    auto _coeff_ref_dispatch(const Tuple &idxs,
                             std::integer_sequence<rank_type, ChildRanks...>)
        -> value_type &
        requires(traits::is_assignable())
    {
        return expression().coeff_ref(
            _get_child_index<ChildRanks>(idxs)...);
    }

    template <typename Tuple, rank_type... ChildRanks>
    auto _const_coeff_ref_dispatch(
        const Tuple &idxs,
        std::integer_sequence<rank_type, ChildRanks...>) const
        -> const value_type &
        requires(traits::is_referrable())
    {
        return expression().const_coeff_ref(
            _get_child_index<ChildRanks>(idxs)...);
    }

    template <typename... Args>
    auto coeff(Args &&...idxs) const -> value_type {
        auto tup = std::make_tuple(std::forward<Args>(idxs)...);
        return _coeff_dispatch(
            tup,
            std::make_integer_sequence<rank_type, child_extents_type::rank()>{});
    }

    template <typename... Args>
    auto coeff_ref(Args &&...idxs) -> value_type &
        requires(traits::is_assignable())
    {
        auto tup = std::make_tuple(std::forward<Args>(idxs)...);
        return _coeff_ref_dispatch(
            tup,
            std::make_integer_sequence<rank_type, child_extents_type::rank()>{});
    }

    template <typename... Args>
    auto const_coeff_ref(Args &&...idxs) const -> const value_type &
        requires(traits::is_referrable())
    {
        auto tup = std::make_tuple(std::forward<Args>(idxs)...);
        return _const_coeff_ref_dispatch(
            tup,
            std::make_integer_sequence<rank_type, child_extents_type::rank()>{});
    }

    template <zipper::concepts::Expression V>
    void assign(const V &v)
        requires(
            traits::is_assignable() &&
            extents_traits::template is_convertable_from<
                typename zipper::expression::detail::ExpressionTraits<
                    V>::extents_type>())
    {
        expression::detail::AssignHelper<V, self_type>::assign(v, *this);
    }

   private:
    /// For a given child rank index, find which output rank maps to it,
    /// then extract that index from the tuple.
    template <rank_type ChildRank, typename Tuple>
    static auto _get_child_index(const Tuple &idxs) -> index_type {
        constexpr rank_type output_pos = _find_output_pos<ChildRank>();
        return std::get<output_pos>(idxs);
    }

    /// Find the output position that maps to a given child rank.
    template <rank_type ChildRank>
    static consteval rank_type _find_output_pos() {
        for (rank_type i = 0; i < output_rank; ++i) {
            if (child_index_map[i] == ChildRank) {
                return i;
            }
        }
        // Should never happen if InsertedDims is valid
        return output_rank;
    }
};

template <rank_type... InsertedDims,
          zipper::concepts::Expression ExprType>
AntiSlice(const ExprType &) -> AntiSlice<const ExprType&, InsertedDims...>;

template <rank_type... InsertedDims,
          zipper::concepts::Expression ExprType>
AntiSlice(ExprType &) -> AntiSlice<ExprType&, InsertedDims...>;

}  // namespace unary
}  // namespace zipper::expression
#endif
