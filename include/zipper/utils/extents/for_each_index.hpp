#if !defined(ZIPPER_UTILS_EXTENTS_FOR_EACH_INDEX_HPP)
#define ZIPPER_UTILS_EXTENTS_FOR_EACH_INDEX_HPP
#include "zipper/detail/LayoutPreference.hpp"
#include "zipper/types.hpp"
#include <array>
#include <utility>

namespace zipper::utils::extents {

// ── Row-major iteration (layout_right) ─────────────────────────────────
//
// For rank-2: outer loop over dim 0 (rows), inner over dim 1 (cols).
// Generalises: dim 0 varies slowest, dim (rank-1) varies fastest.

/// Iterate all index combinations in row-major order (dim 0 slowest).
/// Calls fn(i0, i1, ..., i_{rank-1}) for each valid combination.
template <typename Extents, typename Fn>
void for_each_index_row_major(const Extents &ext, Fn &&fn) {
    constexpr rank_type R = Extents::rank();
    if constexpr (R == 0) {
        fn();
    } else if constexpr (R == 1) {
        for (index_type i = 0; i < ext.extent(0); ++i) { fn(i); }
    } else if constexpr (R == 2) {
        // Row-major: outer = dim 0 (rows), inner = dim 1 (cols).
        for (index_type i = 0; i < ext.extent(0); ++i) {
            for (index_type j = 0; j < ext.extent(1); ++j) { fn(i, j); }
        }
    } else {
        // General rank > 2: iterate dims in natural order (0 slowest).
        std::array<index_type, R> idx{};
        auto iterate = [&](this auto &self, rank_type d) -> void {
            if (d == R) {
                [&]<rank_type... Is>(std::integer_sequence<rank_type, Is...>) {
                    fn(idx[Is]...);
                }(std::make_integer_sequence<rank_type, R>{});
                return;
            }
            for (index_type i = 0; i < ext.extent(d); ++i) {
                idx[d] = i;
                self(d + 1);
            }
        };
        iterate(rank_type{0});
    }
}

// ── Column-major iteration (layout_left) ───────────────────────────────
//
// For rank-2: outer loop over dim 1 (cols), inner over dim 0 (rows).
// Generalises: dim (rank-1) varies slowest, dim 0 varies fastest.

/// Iterate all index combinations in column-major order (last dim slowest).
/// Calls fn(i0, i1, ..., i_{rank-1}) in column-major traversal.
/// Arguments to fn are ALWAYS in natural order (i0, i1, ...) regardless
/// of iteration order.
template <typename Extents, typename Fn>
void for_each_index_col_major(const Extents &ext, Fn &&fn) {
    constexpr rank_type R = Extents::rank();
    if constexpr (R == 0) {
        fn();
    } else if constexpr (R == 1) {
        for (index_type i = 0; i < ext.extent(0); ++i) { fn(i); }
    } else if constexpr (R == 2) {
        // Column-major: outer = dim 1, inner = dim 0.
        // fn called with (row, col) -- natural order.
        for (index_type j = 0; j < ext.extent(1); ++j) {
            for (index_type i = 0; i < ext.extent(0); ++i) { fn(i, j); }
        }
    } else {
        // General rank > 2: iterate dims in reverse order but call fn
        // with natural-order indices. Use an index buffer.
        std::array<index_type, R> idx{};
        auto iterate = [&](this auto &self, rank_type d) -> void {
            // d counts which reversed dimension we're iterating.
            // Actual dimension = R - 1 - d.
            rank_type actual_dim = R - 1 - d;
            if (d == R) {
                [&]<rank_type... Is>(std::integer_sequence<rank_type, Is...>) {
                    fn(idx[Is]...);
                }(std::make_integer_sequence<rank_type, R>{});
                return;
            }
            for (index_type i = 0; i < ext.extent(actual_dim); ++i) {
                idx[actual_dim] = i;
                self(d + 1);
            }
        };
        iterate(rank_type{0});
    }
}

// ── Layout-dispatched iteration ────────────────────────────────────────

/// Dispatch to row-major or column-major based on a layout preference type.
/// NoLayoutPreference and DenseLayoutPreference<layout_right> -> row-major.
/// DenseLayoutPreference<layout_left> -> column-major.
template <typename LayoutPref, typename Extents, typename Fn>
void for_each_index(const Extents &ext, Fn &&fn) {
    constexpr bool is_col_major = [] {
        if constexpr (requires { typename LayoutPref::layout_policy; }) {
            return std::is_same_v<typename LayoutPref::layout_policy,
                                  zipper::storage::layout_left>;
        } else {
            return false; // NoLayoutPreference -> row-major
        }
    }();

    if constexpr (is_col_major) {
        for_each_index_col_major(ext, std::forward<Fn>(fn));
    } else {
        for_each_index_row_major(ext, std::forward<Fn>(fn));
    }
}

} // namespace zipper::utils::extents
#endif
