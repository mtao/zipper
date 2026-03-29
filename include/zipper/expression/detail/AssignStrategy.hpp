#pragma once

// ── AssignStrategy ───────────────────────────────────────────────────────
//
// Strategy types that tell AssignHelper HOW to perform assignment from
// a source expression to a target.
//
// Expressions declare their preferred strategy via:
//   struct ExpressionTraits<MyExpr> {
//       using assign_strategy = FiberAssignStrategy<FiberIndices...>;
//   };
//
// If no assign_strategy is declared, DefaultAssignStrategy is assumed.

#include "zipper/types.hpp"
#include <type_traits>

namespace zipper::expression::detail {

// ── Strategy tag types ─────────────────────────────────────────────────

/// Default element-by-element assignment.
/// Iterates over all index combinations, reading each coefficient from
/// the source and writing to the target. The iteration order is
/// determined by the target's preferred layout (row-major by default).
struct DefaultAssignStrategy {};

/// Fiber-based assignment for expressions like PartialTransform.
///
/// Instead of evaluating the source expression once per coefficient
/// (O(m*n) calls), this strategy iterates over the "non-fiber" indices
/// and evaluates the transform function once per fiber (O(m) calls for
/// an m x n matrix with column fibers).
///
/// FiberIndices... are the dimension indices that form the fiber
/// (e.g., {0} for column fibers = rowwise transform,
///        {1} for row fibers = colwise transform).
///
/// The assignment loop:
///   for each combination of non-fiber indices:
///     result_fiber = fn(source_fiber)
///     write result_fiber to target at those non-fiber indices
///
/// This is naturally alias-safe when fibers don't overlap (which they
/// don't for standard rowwise/colwise transforms).
template <rank_type... FiberIndices>
struct FiberAssignStrategy {};

// ── Detection machinery ────────────────────────────────────────────────

/// Checks whether a traits type declares a custom assign_strategy.
template <typename ET>
concept HasAssignStrategy = requires { typename ET::assign_strategy; };

/// Checks whether a traits type declares a non-default assign_strategy.
template <typename ET>
concept HasCustomAssignStrategy =
    HasAssignStrategy<ET>
    && !std::is_same_v<typename ET::assign_strategy, DefaultAssignStrategy>;

/// Extract the assign_strategy from an ExpressionTraits type, defaulting
/// to DefaultAssignStrategy if none is declared.
template <typename ET, typename = void>
struct get_assign_strategy {
    using type = DefaultAssignStrategy;
};

template <typename ET>
struct get_assign_strategy<ET, std::void_t<typename ET::assign_strategy>> {
    using type = typename ET::assign_strategy;
};

template <typename ET>
using get_assign_strategy_t = typename get_assign_strategy<ET>::type;

// ── FiberAssignStrategy detection ──────────────────────────────────────

template <typename T>
struct is_fiber_assign_strategy : std::false_type {};

template <rank_type... Indices>
struct is_fiber_assign_strategy<FiberAssignStrategy<Indices...>>
  : std::true_type {};

template <typename T>
inline constexpr bool is_fiber_assign_strategy_v =
    is_fiber_assign_strategy<T>::value;

} // namespace zipper::expression::detail
