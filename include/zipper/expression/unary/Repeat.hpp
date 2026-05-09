
#if !defined(ZIPPER_EXPRESSION_UNARY_REPEAT_HPP)
#define ZIPPER_EXPRESSION_UNARY_REPEAT_HPP

#include "UnaryExpressionBase.hpp"
#include "zipper/concepts/Extents.hpp"
#include "zipper/expression/binary/TensorProduct.hpp"
#include "zipper/expression/detail/IndexSet.hpp"

namespace zipper::expression {
namespace unary {
enum class RepeatMode { Left, Right };
template <RepeatMode Mode, rank_type Count,
          zipper::concepts::QualifiedExpression Child>
class Repeat;

namespace _detail_repeat {
template <RepeatMode Mode, rank_type Count, zipper::concepts::Extents ET>
struct RepeatHelper {
    template <size_t... I>
    constexpr static auto make_repeat_extents_type(
        std::integer_sequence<size_t, I...>) {
        return extents<((void)I, dynamic_extent)...>{};
    }
    using repeat_extents_type = std::decay_t<decltype(make_repeat_extents_type(
        std::make_index_sequence<Count>{}))>;

    using left_extents_type =
        std::conditional_t<Mode == RepeatMode::Left, repeat_extents_type, ET>;
    using right_extents_type =
        std::conditional_t<Mode == RepeatMode::Right, repeat_extents_type, ET>;
    static_assert(zipper::concepts::Extents<left_extents_type>);
    static_assert(zipper::concepts::Extents<right_extents_type>);
    using tensor_type =
        typename binary::_detail_tensor::tensor_coeffwise_extents_values<
            left_extents_type, right_extents_type>;

    using extents_type = typename tensor_type::product_extents_type;
    constexpr static rank_type base_rank = ET::rank();
    constexpr static rank_type offset_rank =
        Mode == RepeatMode::Left ? Count : 0;
    constexpr static extents_type make_extents(const ET& e) {
        if constexpr (Mode == RepeatMode::Left) {
            return tensor_type::merge(repeat_extents_type{}, e);
        } else {
            return tensor_type::merge(e, repeat_extents_type{});
        }
    }
};
}  // namespace _detail_repeat

}  // namespace unary

/// Implementation details for Repeat expressions.
///
/// Holds the child's base rank and the offset rank (number of dimensions
/// prepended or appended). The Repeat class body needs these to correctly
/// map output indices to child indices.
template <unary::RepeatMode Mode, rank_type Count,
          zipper::concepts::QualifiedExpression Child>
struct detail::ExpressionDetail<unary::Repeat<Mode, Count, Child>> {
    using BaseTraits = expression::detail::ExpressionTraits<std::decay_t<Child>>;
    using helper_type =
        unary::_detail_repeat::RepeatHelper<Mode, Count,
                                    typename BaseTraits::extents_type>;
    constexpr static rank_type base_rank = helper_type::base_rank;
    constexpr static rank_type offset_rank = helper_type::offset_rank;
};

template <unary::RepeatMode Mode, rank_type Count,
          zipper::concepts::QualifiedExpression Child>
struct detail::ExpressionTraits<unary::Repeat<Mode, Count, Child>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
          Child,
          zipper::detail::AccessFeatures::const_value()> {
    using _Detail = detail::ExpressionDetail<unary::Repeat<Mode, Count, Child>>;
    constexpr static bool is_coefficient_consistent = true;
    constexpr static bool is_value_based = false;
    using extents_type = typename _Detail::helper_type::extents_type;

    /// Propagate has_index_set from child — Repeat broadcasts values but
    /// does not change the child's sparsity structure along child dimensions.
    /// Repeated (broadcast) dimensions are fully dense.
    using child_traits = ExpressionTraits<std::decay_t<Child>>;
    constexpr static bool has_index_set = child_traits::has_index_set;

    /// Backward-compatible alias for has_index_set.
    constexpr static bool has_known_zeros = has_index_set;
};
namespace unary {
template <RepeatMode Mode, rank_type Count,
          zipper::concepts::QualifiedExpression Child>
class Repeat : public UnaryExpressionBase<Repeat<Mode, Count, Child>, Child> {
    //

   public:
    using self_type = Repeat<Mode, Count, Child>;
    using Base = UnaryExpressionBase<Repeat<Mode, Count, Child>, Child>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using detail_type = zipper::expression::detail::ExpressionDetail<self_type>;
    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    constexpr static bool is_static = extents_traits::is_static;
    using value_type = traits::value_type;
    constexpr static rank_type base_rank = detail_type::base_rank;
    constexpr static rank_type pad_offset = detail_type::offset_rank;

    template <typename U>
      requires std::constructible_from<typename Base::storage_type, U &&>
    Repeat(U &&a)
        : Base(std::forward<U>(a)) {}

    /// Recursively deep-copy child so the result owns all data.
    auto make_owned() const {
        auto owned_child = expression().make_owned();
        return Repeat<Mode, Count, const decltype(owned_child)>(
            std::move(owned_child));
    }

    constexpr auto extent(rank_type i) const -> index_type {
        if constexpr (Mode == RepeatMode::Left) {
            if (i < Count) {
                return extents_type::static_extent(i);  // dummy dim (0)
            } else {
                return expression().extent(i - Count);
            }
        } else {
            if (i < base_rank) {
                return expression().extent(i);
            } else {
                return extents_type::static_extent(i);  // dummy dim (0)
            }
        }
    }

    constexpr auto extents() const -> extents_type {
        return extents_traits::make_extents_from(*this);
    }

    using Base::expression;
    template <typename... Args, rank_type... ranks>
    auto get_value(std::integer_sequence<rank_type, ranks...>,
                   Args&&... args) const -> decltype(auto) {
        return expression()(zipper::detail::pack_index<ranks + pad_offset>(
            std::forward<Args>(args)...)...);
    }
    template <typename... Args>
    value_type coeff(Args&&... args) const {
        return get_value(std::make_integer_sequence<rank_type, base_rank>{},
                         std::forward<Args>(args)...);
    }

    // ── Index set propagation ───────────────────────────────────────────
    //
    // Repeated (broadcast) dimensions are fully dense: all indices are
    // active because the value is the same regardless of the index along
    // that dimension.
    //
    // Child dimensions forward to the child's index_set, preserving the
    // child's sparsity structure.  When the child is rank-1 (promoted to
    // rank-2 via Repeat), the child's no-arg index_set<0>() is used
    // because the sparsity pattern is independent of the broadcast index.

    /// @brief Index set for dimension @p D (rank-2+ output).
    ///
    /// Returns FullRange for repeated dimensions and forwards to the child
    /// for child dimensions.
    template <rank_type D>
      requires(traits::has_index_set && D < extents_type::rank() &&
               extents_type::rank() >= 2)
    auto index_set(index_type other_idx) const {
        if constexpr (Mode == RepeatMode::Left) {
            if constexpr (D < Count) {
                // Repeated dimension: all indices active.
                return zipper::expression::detail::FullRange{extent(D)};
            } else {
                constexpr rank_type child_dim = D - Count;
                if constexpr (base_rank == 1) {
                    // Child is rank-1: index_set takes no args.
                    return expression().template index_set<child_dim>();
                } else {
                    return expression().template index_set<child_dim>(
                        other_idx);
                }
            }
        } else {  // RepeatMode::Right
            if constexpr (D < base_rank) {
                constexpr rank_type child_dim = D;
                if constexpr (base_rank == 1) {
                    return expression().template index_set<child_dim>();
                } else {
                    return expression().template index_set<child_dim>(
                        other_idx);
                }
            } else {
                // Repeated dimension: all indices active.
                return zipper::expression::detail::FullRange{extent(D)};
            }
        }
    }

    /// @brief Index set for dimension @p D (rank-1 output, Count == 0).
    template <rank_type D>
      requires(traits::has_index_set && D == 0 &&
               extents_type::rank() == 1)
    auto index_set() const {
        return expression().template index_set<0>();
    }

    /// @deprecated Use index_set instead.
    template <rank_type D>
      requires(traits::has_index_set && D < extents_type::rank() &&
               extents_type::rank() >= 2)
    auto nonzero_range(index_type other_idx) const {
        return index_set<D>(other_idx);
    }

    /// @deprecated Use index_set instead.
    template <rank_type D>
      requires(traits::has_index_set && D == 0 &&
               extents_type::rank() == 1)
    auto nonzero_range() const {
        return index_set<D>();
    }

    /// Convenience: col_range_for_row (rank-2 output only).
    auto col_range_for_row(index_type row) const
      requires(traits::has_index_set && extents_type::rank() == 2)
    {
        return index_set<1>(row);
    }

    /// Convenience: row_range_for_col (rank-2 output only).
    auto row_range_for_col(index_type col) const
      requires(traits::has_index_set && extents_type::rank() == 2)
    {
        return index_set<0>(col);
    }

    /// Convenience: nonzero_segment (rank-1 output only).
    auto nonzero_segment() const
      requires(traits::has_index_set && extents_type::rank() == 1)
    {
        return index_set<0>();
    }

};

}  // namespace unary
}  // namespace zipper::expression

#endif
