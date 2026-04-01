
#if !defined(ZIPPER_EXPRESSION_UNARY_LIFT_HPP)
#define ZIPPER_EXPRESSION_UNARY_LIFT_HPP

#include "UnaryExpressionBase.hpp"
#include "zipper/concepts/Extents.hpp"
#include "zipper/expression/binary/TensorProduct.hpp"
#include "zipper/expression/detail/IndexSet.hpp"

namespace zipper::expression {
namespace unary {

template <rank_type Count, zipper::concepts::QualifiedExpression Child>
class Lift;

namespace _detail_lift {
/// Compute the extended extents for Lift: append @p Count dynamic dimensions
/// after the child's existing extents.
template <rank_type Count, zipper::concepts::Extents ET>
struct LiftHelper {
    template <size_t... I>
    constexpr static auto make_pad_extents_type(
        std::integer_sequence<size_t, I...>) {
        return extents<((void)I, dynamic_extent)...>{};
    }
    using pad_extents_type = std::decay_t<decltype(make_pad_extents_type(
        std::make_index_sequence<Count>{}))>;

    using tensor_type =
        typename binary::_detail_tensor::tensor_coeffwise_extents_values<
            ET, pad_extents_type>;

    using extents_type = typename tensor_type::product_extents_type;
    constexpr static rank_type base_rank = ET::rank();

    constexpr static extents_type make_extents(const ET& e) {
        return tensor_type::merge(e, pad_extents_type{});
    }
};
}  // namespace _detail_lift

}  // namespace unary

/// ExpressionDetail for Lift: holds base_rank of the child.
template <rank_type Count, zipper::concepts::QualifiedExpression Child>
struct detail::ExpressionDetail<unary::Lift<Count, Child>> {
    using BaseTraits = expression::detail::ExpressionTraits<std::decay_t<Child>>;
    using helper_type =
        unary::_detail_lift::LiftHelper<Count,
                                        typename BaseTraits::extents_type>;
    constexpr static rank_type base_rank = helper_type::base_rank;
};

template <rank_type Count, zipper::concepts::QualifiedExpression Child>
struct detail::ExpressionTraits<unary::Lift<Count, Child>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
          Child,
          zipper::detail::AccessFeatures::const_value()> {
    using _Detail = detail::ExpressionDetail<unary::Lift<Count, Child>>;
    constexpr static bool is_coefficient_consistent = true;
    constexpr static bool is_value_based = false;
    using extents_type = typename _Detail::helper_type::extents_type;

    /// Propagate has_index_set from child — Lift broadcasts values but
    /// does not change the child's sparsity structure along child dimensions.
    /// Lifted (appended) dimensions are fully dense.
    using child_traits = ExpressionTraits<std::decay_t<Child>>;
    constexpr static bool has_index_set = child_traits::has_index_set;

    /// Backward-compatible alias for has_index_set.
    constexpr static bool has_known_zeros = has_index_set;
};

namespace unary {
template <rank_type Count, zipper::concepts::QualifiedExpression Child>
class Lift : public UnaryExpressionBase<Lift<Count, Child>, Child> {
   public:
    using self_type = Lift<Count, Child>;
    using Base = UnaryExpressionBase<Lift<Count, Child>, Child>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using detail_type = zipper::expression::detail::ExpressionDetail<self_type>;
    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    constexpr static bool is_static = extents_traits::is_static;
    using value_type = traits::value_type;
    constexpr static rank_type base_rank = detail_type::base_rank;

    template <typename U>
      requires std::constructible_from<typename Base::storage_type, U &&>
    Lift(U &&a)
        : Base(std::forward<U>(a)) {}

    /// Recursively deep-copy child so the result owns all data.
    auto make_owned() const {
        auto owned_child = expression().make_owned();
        return Lift<Count, const decltype(owned_child)>(
            std::move(owned_child));
    }

    constexpr auto extent(rank_type i) const -> index_type {
        if (i < base_rank) {
            return expression().extent(i);
        } else {
            return extents_type::static_extent(i);  // appended dim (dynamic)
        }
    }

    constexpr auto extents() const -> extents_type {
        return extents_traits::make_extents_from(*this);
    }

    using Base::expression;

    /// Extract the child's indices from the full argument pack.
    /// Since appended dims come after child dims, we just take
    /// the first base_rank arguments.
    template <typename... Args, rank_type... ranks>
    auto get_value(std::integer_sequence<rank_type, ranks...>,
                   Args&&... args) const -> decltype(auto) {
        return expression()(zipper::detail::pack_index<ranks>(
            std::forward<Args>(args)...)...);
    }

    template <typename... Args>
    value_type coeff(Args&&... args) const {
        return get_value(std::make_integer_sequence<rank_type, base_rank>{},
                         std::forward<Args>(args)...);
    }

    // ── Index set propagation ───────────────────────────────────────────
    //
    // Appended (lifted) dimensions are fully dense: all indices are
    // active because the value is the same regardless of the index along
    // that dimension.
    //
    // Child dimensions forward to the child's index_set, preserving the
    // child's sparsity structure.  When the child is rank-1 (lifted to
    // rank-2 via Lift), the child's no-arg index_set<0>() is used
    // because the sparsity pattern is independent of the broadcast index.

    /// @brief Index set for dimension @p D (rank-2+ output).
    ///
    /// Returns FullRange for lifted dimensions and forwards to the child
    /// for child dimensions.
    template <rank_type D>
      requires(traits::has_index_set && D < extents_type::rank() &&
               extents_type::rank() >= 2)
    auto index_set(index_type other_idx) const {
        if constexpr (D < base_rank) {
            constexpr rank_type child_dim = D;
            if constexpr (base_rank == 1) {
                return expression().template index_set<child_dim>();
            } else {
                return expression().template index_set<child_dim>(
                    other_idx);
            }
        } else {
            // Lifted dimension: all indices active.
            return zipper::expression::detail::FullRange{extent(D)};
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
