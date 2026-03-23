#if !defined(ZIPPER_STORAGE_DETAIL_TO_SPARSE_COMPRESSED_DATA_HPP)
#define ZIPPER_STORAGE_DETAIL_TO_SPARSE_COMPRESSED_DATA_HPP

#include "SparseCompressedData.hpp"
#include "zipper/storage/SparseCoordinateAccessor.hpp"
#include "zipper/storage/layout_types.hpp"

#include <algorithm>
#include <numeric>

namespace zipper::storage::detail {

/// Convert a SparseCoordinateAccessor to SparseCompressedData (SOA format).
///
/// @tparam LayoutPolicy  `layout_right` (CSR, default) stores indices in
///                       natural order (row, col, ...).
///                       `layout_left` (CSC) reverses the index order so
///                       the compressed data is stored column-major.
///
/// For rank-2 matrices, this produces standard CSR/CSC arrays:
///   m_indptr[n_outer + 1]   — row/column pointer array
///   m_indices[nnz]          — column/row indices
///   m_values[nnz]           — nonzero values
///
/// For rank-1 vectors:
///   m_indices[nnz]          — sorted nonzero indices
///   m_values[nnz]           — corresponding values
template <typename LayoutPolicy = default_layout_policy,
          typename ValueType = double,
          typename Extents = dextents<0>()>
SparseCompressedData<ValueType, Extents::rank() - 1> to_sparse_compressed_data(
    const SparseCoordinateAccessor<ValueType, Extents>& e) {
    constexpr bool ReverseIndices =
        std::is_same_v<LayoutPolicy, layout_left>;
    constexpr auto Rank = Extents::rank();

    if (!e.is_compressed()) {
        SparseCoordinateAccessor<ValueType, Extents> e2 = e;
        e2.compress();
        return to_sparse_compressed_data<LayoutPolicy>(e2);
    }

    SparseCompressedData<ValueType, Rank - 1> R;

    if constexpr (Rank == 1) {
        // Rank-1 (vector): just copy sorted indices and values.
        size_t count = static_cast<size_t>(std::distance(e.begin(), e.end()));
        R.m_indices.reserve(count);
        R.m_values.reserve(count);
        for (const auto& v : e) {
            R.m_indices.push_back(std::get<0>(v.multiindex()));
            R.m_values.push_back(v.value());
        }
    } else if constexpr (Rank == 2) {
        // Rank-2 (matrix): build standard CSR/CSC with dense indptr.
        //
        // For CSR (natural order):  outer = dim 0 (rows), inner = dim 1 (cols)
        // For CSC (reversed order): outer = dim 1 (cols), inner = dim 0 (rows)

        // Determine outer dimension size from the extents.
        // For CSR: n_outer = extent(0). For CSC: n_outer = extent(1).
        index_type n_outer = ReverseIndices ? e.extent(1) : e.extent(0);

        // Collect all entries with (outer, inner, value)
        struct Entry {
            index_type outer;
            index_type inner;
            ValueType value;
        };
        std::vector<Entry> entries;
        size_t count = static_cast<size_t>(std::distance(e.begin(), e.end()));
        entries.reserve(count);
        for (const auto& v : e) {
            auto mi = v.multiindex();
            auto row = std::get<0>(mi);
            auto col = std::get<1>(mi);
            if constexpr (ReverseIndices) {
                entries.push_back({col, row, v.value()});
            } else {
                entries.push_back({row, col, v.value()});
            }
        }

        // Sort by (outer, inner) — CSR entries from COO are already in
        // row-major order, but CSC needs re-sorting by column.
        if constexpr (ReverseIndices) {
            std::sort(entries.begin(), entries.end(),
                [](const Entry& a, const Entry& b) {
                    if (a.outer != b.outer) return a.outer < b.outer;
                    return a.inner < b.inner;
                });
        }
        // For CSR (non-reversed), COO compress() already sorts row-major,
        // so entries are in the correct order.

        size_t nnz = entries.size();

        // Build indptr by counting entries per outer index
        R.m_indptr.assign(n_outer + 1, index_type(0));
        for (const auto& entry : entries) {
            assert(entry.outer < n_outer);
            R.m_indptr[entry.outer + 1]++;
        }
        // Prefix sum
        for (index_type j = 1; j <= n_outer; ++j) {
            R.m_indptr[j] += R.m_indptr[j - 1];
        }

        // Fill indices and values
        R.m_indices.resize(nnz);
        R.m_values.resize(nnz);
        // Since entries are already sorted by (outer, inner), we can just
        // copy them sequentially.
        for (size_t k = 0; k < nnz; ++k) {
            R.m_indices[k] = entries[k].inner;
            R.m_values[k] = entries[k].value;
        }
    } else {
        // Rank >= 3: use insert_back for general recursive construction.
        // This is the uncommon path (only used in tests).
        auto m = [&R]<rank_type... Ranks>(
                     std::integer_sequence<rank_type, Ranks...>,
                     const auto& t) -> auto& {
            if constexpr (ReverseIndices) {
                return R.insert_back(
                    std::get<Rank - 1 - Ranks>(t)...);
            } else {
                return R.insert_back(std::get<Ranks>(t)...);
            }
        };

        if constexpr (ReverseIndices) {
            // Re-sort by reversed index order
            struct Entry {
                std::array<index_type, Rank> indices;
                ValueType value;
            };
            std::vector<Entry> entries;
            entries.reserve(static_cast<size_t>(
                std::distance(e.begin(), e.end())));
            for (const auto& v : e) {
                Entry entry;
                auto mi = v.multiindex();
                [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                    ((entry.indices[Is] = std::get<Is>(mi)), ...);
                }(std::make_index_sequence<Rank>{});
                entry.value = v.value();
                entries.push_back(entry);
            }

            std::sort(entries.begin(), entries.end(),
                [](const Entry& a, const Entry& b) {
                    for (int d = Rank - 1; d >= 0; --d) {
                        if (a.indices[d] < b.indices[d]) return true;
                        if (a.indices[d] > b.indices[d]) return false;
                    }
                    return false;
                });

            for (const auto& entry : entries) {
                auto insert = [&R]<rank_type... Ranks>(
                    std::integer_sequence<rank_type, Ranks...>,
                    const std::array<index_type, Rank>& idx)
                    -> auto& {
                    return R.insert_back(
                        idx[Rank - 1 - Ranks]...);
                };
                insert(
                    std::make_integer_sequence<rank_type, Rank>{},
                    entry.indices) = entry.value;
            }
        } else {
            for (const auto& v : e) {
                m(std::make_integer_sequence<rank_type, Rank>{},
                  v.multiindex()) = v.value();
            }
        }
    }
    return R;
}

/// Overload that also accepts the outer dimension size explicitly.
/// Useful when the SparseCoordinateAccessor has dynamic extents and
/// the caller knows the correct outer size.
template <typename LayoutPolicy = default_layout_policy,
          typename ValueType = double,
          typename Extents = dextents<0>()>
SparseCompressedData<ValueType, Extents::rank() - 1>
to_sparse_compressed_data(
    const SparseCoordinateAccessor<ValueType, Extents>& e,
    index_type n_outer) {
    auto R = to_sparse_compressed_data<LayoutPolicy>(e);
    // Ensure indptr is sized correctly for the given n_outer
    if constexpr (Extents::rank() >= 2) {
        if (R.m_indptr.size() < n_outer + 1) {
            R.m_indptr.resize(n_outer + 1, R.m_indptr.empty() ? 0 : R.m_indptr.back());
        }
    }
    return R;
}

}  // namespace zipper::storage::detail
#endif
