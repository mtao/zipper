#if !defined(ZIPPER_EXPRESSION_DETAIL_ASSIGNHELPER_HPP)
#define ZIPPER_EXPRESSION_DETAIL_ASSIGNHELPER_HPP
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/ExtentsTraits.hpp"
#include "zipper/detail/assert.hpp"
#include "zipper/expression/detail/AssignStrategy.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"
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
    requires(ExpressionTraits<To>::is_writable
             && zipper::detail::ExtentsTraits<
                 typename ExpressionTraits<To>::extents_type>::
                 template is_convertable_from<
                     typename ExpressionTraits<From>::extents_type>())
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

    /// Element-by-element copy, iterating in the target's preferred layout.
    static void assign_direct(const From &from, To &to);

    /// Main entry point: handles resizing, aliasing, and strategy dispatch.
    static void assign(const From &from, To &to);
};

template <zipper::concepts::Expression From, zipper::concepts::Expression To>
    requires(ExpressionTraits<To>::is_writable
             && zipper::detail::ExtentsTraits<
                 typename ExpressionTraits<To>::extents_type>::
                 template is_convertable_from<
                     typename ExpressionTraits<From>::extents_type>())
void AssignHelper<From, To>::assign_direct(const From &from, To &to) {
    if constexpr (extents_type::rank() == 0) {
        to() = from();
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
    requires(ExpressionTraits<To>::is_writable
             && zipper::detail::ExtentsTraits<
                 typename ExpressionTraits<To>::extents_type>::
                 template is_convertable_from<
                     typename ExpressionTraits<From>::extents_type>())
void AssignHelper<From, To>::assign(const From &from, To &to) {
    using FromTraits = zipper::expression::detail::ExpressionTraits<From>;
    using ToTraits = zipper::expression::detail::ExpressionTraits<To>;
    constexpr static bool assigning_from_infinite =
        FromTraits::extents_type::rank() == 0;
    constexpr static bool should_resize =
        !assigning_from_infinite && ToTraits::is_resizable();

    // ── Strategy dispatch ────────────────────────────────────────────
    //
    // If the source expression declares a custom assign strategy AND
    // provides an assign_to() method, use it. This allows expressions
    // like PartialTransform to perform fiber-by-fiber assignment
    // (calling fn once per fiber instead of once per element).
    //
    // Custom strategies are also inherently alias-safe (for fiber
    // strategies, fibers don't overlap), so we skip the aliasing
    // temporary path.
    if constexpr (HasCustomAssignStrategy<FromTraits>
                  && HasAssignTo<From, To>) {
        if constexpr (should_resize) {
            to.resize(to_extents_traits::convert_from(from.extents()));
        } else if constexpr (to_extents_traits::is_dynamic
                             && !assigning_from_infinite) {
            ZIPPER_ASSERT(to.extents() == from.extents());
        }
        from.assign_to(to);
    } else if constexpr (get_is_coefficient_consistent<FromTraits>()) {
        // ── Default path: coefficient-consistent (no aliasing) ─────────
        if constexpr (should_resize) {
            to.resize(to_extents_traits::convert_from(from.extents()));
        } else if constexpr (to_extents_traits::is_dynamic
                             && !assigning_from_infinite) {
            ZIPPER_ASSERT(to.extents() == from.extents());
        }

        assign_direct(from, to);
    } else {
        // ── Aliasing path: materialize to temporary first ──────────────
        using POS = nullary::
            MDArray<value_type, extents_type, layout_policy, accessor_policy>;
        POS pos(to_extents_traits::convert_from(from.extents()));

        AssignHelper<From, POS>::assign_direct(from, pos);
        if constexpr (should_resize) {
            to.resize(to_extents_traits::convert_from(from.extents()));
        }
        AssignHelper<POS, To>::assign_direct(pos, to);
    }
}
} // namespace zipper::expression::detail
#endif
