#pragma once

/// @file NonzeroRange.hpp
/// @brief Lightweight range types for describing non-zero index patterns.
/// @ingroup sparsity
///
/// Expressions with known structural zeros (TriangularView, Identity, Unit,
/// etc.) report their non-zero index patterns via `nonzero_range<D>(...)`.
/// The return type is expression-specific — each expression returns the most
/// efficient range type for its sparsity structure.
///
/// All range types satisfy the `NonzeroRange` concept and provide:
///   - `contains(idx)` — O(1) membership test (O(log n) for SparseIndexRange)
///   - `begin()` / `end()` — forward iteration over non-zero indices
///   - `size()` — number of non-zero indices
///   - `empty()` — whether the range is empty
///
/// Range types:
///   - `ContiguousIndexRange` — half-open interval [first, last)
///   - `SingleIndexRange`     — exactly one index
///   - `SparseIndexRange`     — sorted set of indices
///   - `FullRange`            — all indices in [0, extent)
///
/// @see zipper::expression::unary::TriangularView — returns
///      ContiguousIndexRange from nonzero_range (triangular regions).
/// @see zipper::expression::nullary::Identity — returns SingleIndexRange
///      from nonzero_range (one non-zero per row/column).
/// @see zipper::expression::nullary::Unit — returns SingleIndexRange
///      from nonzero_range (one non-zero in a rank-1 expression).
/// @see zipper::expression::binary::ZeroAwareOperation — propagates
///      nonzero_range through addition/subtraction (union semantics).
/// @see zipper::expression::binary::MatrixProduct — uses nonzero_range
///      for intersection semantics in dot product loops.
/// @see zipper::expression::binary::MatrixVectorProduct — uses nonzero_range
///      for intersection semantics in matrix-vector dot product loops.

#include <algorithm>
#include <ranges>
#include <vector>

#include "zipper/types.hpp"

namespace zipper::expression::detail {

// ─────────────────────────────────────────────────────────────────────────────
/// @brief A contiguous half-open interval [first, last) of indices.
///
/// Used by TriangularView and other expressions whose non-zero columns in a
/// given row form a contiguous block.
// ─────────────────────────────────────────────────────────────────────────────
struct ContiguousIndexRange {
    index_type first;  ///< Inclusive lower bound.
    index_type last;   ///< Exclusive upper bound.

    /// @brief Test whether @p idx is in the range [first, last).
    constexpr auto contains(index_type idx) const -> bool {
        return idx >= first && idx < last;
    }

    constexpr auto empty() const -> bool { return first >= last; }
    constexpr auto size() const -> index_type {
        return empty() ? index_type{0} : last - first;
    }

    auto begin() const { return std::ranges::iota_view(first, last).begin(); }
    auto end() const { return std::ranges::iota_view(first, last).end(); }
};

// ─────────────────────────────────────────────────────────────────────────────
/// @brief A range containing exactly one index.
///
/// Used by Identity (rank-2) and Diagonal expressions, where each row has
/// exactly one non-zero column.
// ─────────────────────────────────────────────────────────────────────────────
struct SingleIndexRange {
    index_type value;  ///< The single non-zero index.

    /// @brief Test whether @p idx equals the stored value.
    constexpr auto contains(index_type idx) const -> bool {
        return idx == value;
    }

    constexpr auto empty() const -> bool { return false; }
    constexpr auto size() const -> index_type { return 1; }

    auto begin() const {
        return std::ranges::iota_view(value, value + 1).begin();
    }
    auto end() const {
        return std::ranges::iota_view(value, value + 1).end();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
/// @brief A sorted set of non-zero indices for general sparse patterns.
///
/// The indices must be stored in ascending order so that `intersect_nonzeros`
/// (and other merge-based algorithms) work correctly.
// ─────────────────────────────────────────────────────────────────────────────
struct SparseIndexRange {
    std::vector<index_type> indices;  ///< Sorted ascending.

    /// @brief Test whether @p idx is present (binary search, O(log n)).
    auto contains(index_type idx) const -> bool {
        return std::binary_search(indices.begin(), indices.end(), idx);
    }

    auto empty() const -> bool { return indices.empty(); }
    auto size() const -> index_type { return indices.size(); }

    auto begin() const { return indices.begin(); }
    auto end() const { return indices.end(); }
};

// ─────────────────────────────────────────────────────────────────────────────
/// @brief The full index range [0, extent) — every position is non-zero.
///
/// Used as the default for dense expressions, enabling universal code paths
/// without `if constexpr` branching when desired.
// ─────────────────────────────────────────────────────────────────────────────
struct FullRange {
    index_type extent;  ///< Number of indices: range is [0, extent).

    /// @brief Always returns true (no known zeros).
    constexpr auto contains([[maybe_unused]] index_type idx) const -> bool {
        return true;
    }

    constexpr auto empty() const -> bool { return extent == 0; }
    constexpr auto size() const -> index_type { return extent; }

    auto begin() const {
        return std::ranges::iota_view(index_type{0}, extent).begin();
    }
    auto end() const {
        return std::ranges::iota_view(index_type{0}, extent).end();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
/// @brief Concept satisfied by all non-zero range types.
///
/// Any type that provides `contains(idx)`, forward iteration via
/// `begin()`/`end()`, and `size()`/`empty()` satisfies this concept.
// ─────────────────────────────────────────────────────────────────────────────
template <typename R>
concept NonzeroRange = requires(const R &r, index_type idx) {
    { r.contains(idx) } -> std::convertible_to<bool>;
    { r.begin() };
    { r.end() };
    { r.size() } -> std::convertible_to<index_type>;
    { r.empty() } -> std::convertible_to<bool>;
};

// Static assertions to verify all range types satisfy the concept.
static_assert(NonzeroRange<ContiguousIndexRange>);
static_assert(NonzeroRange<SingleIndexRange>);
static_assert(NonzeroRange<SparseIndexRange>);
static_assert(NonzeroRange<FullRange>);

// ─────────────────────────────────────────────────────────────────────────────
/// @brief Trait to mark unary functors that preserve zero: Op(0) == 0.
///
/// Specialize `is_zero_preserving_unary_op<Op>` for types where this holds.
/// By default, `std::negate` is zero-preserving.
// ─────────────────────────────────────────────────────────────────────────────
template <typename Op>
struct is_zero_preserving_unary_op : std::false_type {};

// std::negate preserves zero: -0 == 0
template <typename T>
struct is_zero_preserving_unary_op<std::negate<T>> : std::true_type {};

template <typename Op>
concept ZeroPreservingUnaryOp = is_zero_preserving_unary_op<Op>::value;

// ─────────────────────────────────────────────────────────────────────────────
/// @brief Trait to mark binary scalar functors that preserve zero when
///        applied with a fixed scalar: Op(0, s) == 0 or Op(s, 0) == 0.
///
/// The template parameters are:
///   - Op: the binary functor type
///   - ScalarOnRight: true if the scalar is the second argument
///
/// By default, `std::multiplies` and `std::divides` (with scalar on right)
/// are zero-preserving.
// ─────────────────────────────────────────────────────────────────────────────
template <typename Op, bool ScalarOnRight>
struct is_zero_preserving_scalar_op : std::false_type {};

// multiplies: 0 * s == 0, s * 0 == 0
template <typename T, bool ScalarOnRight>
struct is_zero_preserving_scalar_op<std::multiplies<T>, ScalarOnRight>
    : std::true_type {};

// divides: 0 / s == 0 (scalar on right only)
template <typename T>
struct is_zero_preserving_scalar_op<std::divides<T>, /*ScalarOnRight=*/true>
    : std::true_type {};

template <typename Op, bool ScalarOnRight>
concept ZeroPreservingScalarOp =
    is_zero_preserving_scalar_op<Op, ScalarOnRight>::value;

}  // namespace zipper::expression::detail
