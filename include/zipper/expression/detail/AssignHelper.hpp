#if !defined(ZIPPER_EXPRESSION_DETAIL_ASSIGNHELPER_HPP)
#define ZIPPER_EXPRESSION_DETAIL_ASSIGNHELPER_HPP
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/ExtentsTraits.hpp"
#include "zipper/detail/assert.hpp"
#include "zipper/expression/concepts/capabilities.hpp"
#include "zipper/expression/detail/AssignStrategy.hpp"
#include "zipper/expression/detail/AssignmentSafety.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/expression/detail/RankZeroEvaluation.hpp"
#include "zipper/utils/extents/assignable_extents.hpp"
#include "zipper/utils/extents/for_each_index.hpp"
#include <tuple>

namespace zipper::expression::nullary {
template <typename ElementType,
          typename Extents,
          typename LayoutPolicy,
          typename AccessorPolicy>
class MDArray;
} // namespace zipper::expression::nullary

namespace zipper::expression::detail {

// ── Detect whether a From expression provides assign_to(To&) ───────────
//
// Expressions with custom assign strategies (e.g. PartialTransform with
// FiberAssignStrategy) can provide an assign_to() method that performs
// optimized assignment. AssignHelper checks for this before falling back
// to element-by-element copying.
template <typename From, typename To>
concept HasAssignTo = requires(const From &from, To &to) {
    { from.assign_to(to) };
};

template <zipper::concepts::Expression From, zipper::concepts::Expression To>
    requires(concepts::WritableExpression<To>
             && !std::is_const_v<std::remove_reference_t<To>>
             && zipper::utils::extents::assignable_extents_v<
                 typename ExpressionTraits<From>::extents_type,
                 typename ExpressionTraits<To>::extents_type>)
struct AssignHelper {
    using to_traits = ExpressionTraits<To>;
    using from_traits = ExpressionTraits<From>;
    using to_extents_type = typename to_traits::extents_type;
    using from_extents_type = typename from_traits::extents_type;

    // "the output sorta traits"
    using traits = to_traits;
    using value_type = typename to_traits::value_type;
    using extents_type = to_extents_type;

    using layout_policy = zipper::default_layout_policy;
    using accessor_policy = zipper::default_accessor_policy<value_type>;

    using to_extents_traits = zipper::detail::ExtentsTraits<to_extents_type>;
    using from_extents_traits =
        zipper::detail::ExtentsTraits<from_extents_type>;

    static constexpr bool scalar_broadcast = from_extents_type::rank() == 0
        && RankZeroEvaluation<From, to_extents_type::rank()>::is_scalar;

    /// Element-by-element copy, iterating in the target's preferred layout.
    static void assign_direct(const From &from, To &to);

    /// Evaluate into a prepared target; the caller handles shape and aliasing.
    static void evaluate_to(const From &from, To &to);

    /// Prepare shape and evaluate; the caller guarantees storage independence.
    static void assign_independent(const From &from, To &to);

    /// Main entry point: handles resizing, aliasing, and strategy dispatch.
    static void assign(const From &from, To &to);
};

template <zipper::concepts::Expression From, zipper::concepts::Expression To>
    requires(concepts::WritableExpression<To>
             && !std::is_const_v<std::remove_reference_t<To>>
             && zipper::utils::extents::assignable_extents_v<
                 typename ExpressionTraits<From>::extents_type,
                 typename ExpressionTraits<To>::extents_type>)
void AssignHelper<From, To>::assign_direct(const From &from, To &to) {
    if constexpr (extents_type::rank() == 0) {
        to() = from();
    } else if constexpr (scalar_broadcast) {
        // Convert proxies to the destination value before the first write.
        const value_type value = from();
        using target_layout_pref = typename to_traits::preferred_layout;
        zipper::utils::extents::for_each_index<target_layout_pref>(
            to.extents(), [&](auto... idxs) { to(idxs...) = value; });
    } else {
        // Use layout-aware iteration: respect the target's preferred layout
        // for cache-friendly traversal order. NoLayoutPreference and
        // layout_right both produce row-major order (same as the old
        // all_extents_indices path), so this is backward-compatible.
        using target_layout_pref = typename to_traits::preferred_layout;
        zipper::utils::extents::for_each_index<target_layout_pref>(
            to.extents(), [&](auto... idxs) { to(idxs...) = from(idxs...); });
    }
}

template <zipper::concepts::Expression From, zipper::concepts::Expression To>
    requires(concepts::WritableExpression<To>
             && !std::is_const_v<std::remove_reference_t<To>>
             && zipper::utils::extents::assignable_extents_v<
                 typename ExpressionTraits<From>::extents_type,
                 typename ExpressionTraits<To>::extents_type>)
void AssignHelper<From, To>::evaluate_to(const From &from, To &to) {
    if constexpr (HasCustomAssignStrategy<from_traits>
                  && HasAssignTo<From, To>) {
        from.assign_to(to);
    } else {
        assign_direct(from, to);
    }
}

template <zipper::concepts::Expression From, zipper::concepts::Expression To>
    requires(concepts::WritableExpression<To>
             && !std::is_const_v<std::remove_reference_t<To>>
             && zipper::utils::extents::assignable_extents_v<
                 typename ExpressionTraits<From>::extents_type,
                 typename ExpressionTraits<To>::extents_type>)
void AssignHelper<From, To>::assign_independent(const From &from, To &to) {
    if constexpr (from_extents_type::rank() != 0) {
        const auto extents = to_extents_traits::convert_from(from.extents());
        if constexpr (to_traits::is_resizable()) {
            to.resize(extents);
        } else {
            ZIPPER_ASSERT(to.extents() == extents);
        }
    }
    evaluate_to(from, to);
}

template <zipper::concepts::Expression From, zipper::concepts::Expression To>
    requires(concepts::WritableExpression<To>
             && !std::is_const_v<std::remove_reference_t<To>>
             && zipper::utils::extents::assignable_extents_v<
                 typename ExpressionTraits<From>::extents_type,
                 typename ExpressionTraits<To>::extents_type>)
void AssignHelper<From, To>::assign(const From &from, To &to) {
    using FromTraits = zipper::expression::detail::ExpressionTraits<From>;
    using ToTraits = zipper::expression::detail::ExpressionTraits<To>;
    constexpr static bool assigning_from_infinite =
        FromTraits::extents_type::rank() == 0;
    constexpr static bool should_resize =
        !assigning_from_infinite && ToTraits::is_resizable();

    // A pointwise proof covers coefficient evaluation, not a custom evaluator's
    // write order. Prove against live storage before any resize or mutation.
    if constexpr (!HasCustomAssignStrategy<FromTraits>) {
        if (assignment_is_safe(from, to)) {
            evaluate_to(from, to);
            return;
        }
    }

    if constexpr (scalar_broadcast && !HasCustomAssignStrategy<FromTraits>) {
        assign_direct(from, to);
    } else {
        // This is the dense assignment coordinator. Sparse storage assignment
        // keeps its own support-aware path through SparseAssignHelper.
        // Evaluate before any destination resize or write, including callbacks
        // that return lazy expressions reading across/within source fibers.
        const auto snapshot_extents = [&] {
            if constexpr (assigning_from_infinite) {
                // Generators and custom evaluators need destination coordinates,
                // not a rank-zero temporary. Never resize for rank-zero input.
                return to.extents();
            } else {
                return to_extents_traits::convert_from(from.extents());
            }
        }();
        if constexpr (!should_resize && to_extents_traits::is_dynamic) {
            ZIPPER_ASSERT(to.extents() == snapshot_extents);
        }
        using POS = nullary::
            MDArray<value_type, extents_type, layout_policy, accessor_policy>;
        POS pos(snapshot_extents);

        AssignHelper<From, POS>::evaluate_to(from, pos);
        if constexpr (should_resize) {
            to.resize(snapshot_extents);
        }
        AssignHelper<POS, To>::assign_direct(pos, to);
    }
}
} // namespace zipper::expression::detail
#endif
