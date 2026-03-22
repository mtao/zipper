#if !defined(ZIPPER_STORAGE_DETAIL_TO_SPARSE_COMPRESSED_DATA_HPP)
#define ZIPPER_STORAGE_DETAIL_TO_SPARSE_COMPRESSED_DATA_HPP

#include "SparseCompressedData.hpp"
#include "zipper/storage/SparseCoordinateAccessor.hpp"
#include "zipper/storage/layout_types.hpp"

namespace zipper::storage::detail {

/// Convert a SparseCoordinateAccessor to SparseCompressedData.
///
/// @tparam LayoutPolicy  `layout_right` (CSR, default) stores indices in
///                       natural order (row, col, ...).
///                       `layout_left` (CSC) reverses the index order so
///                       the compressed data is stored column-major.
///
/// The old `ReverseIndices` bool template parameter is replaced by
/// `LayoutPolicy` for consistency with the rest of the sparse system.
template <typename LayoutPolicy = default_layout_policy,
          typename ValueType = double,
          typename Extents = dextents<0>()>
SparseCompressedData<ValueType, Extents::rank() - 1> to_sparse_compressed_data(
    const SparseCoordinateAccessor<ValueType, Extents>& e) {
    constexpr bool ReverseIndices =
        std::is_same_v<LayoutPolicy, layout_left>;

    if (!e.is_compressed()) {
        SparseCoordinateAccessor<ValueType, Extents> e2 = e;
        e2.compress();
        return to_sparse_compressed_data<LayoutPolicy>(e2);
    } else {
        SparseCompressedData<ValueType, Extents::rank() - 1> R;

        auto m = [&R]<rank_type... Ranks>(
                     std::integer_sequence<rank_type, Ranks...>,
                     const auto& t) -> auto& {
            if constexpr (ReverseIndices) {
                return R.insert_back(
                    std::get<Extents::rank() - 1 - Ranks>(t)...);
            } else {
                return R.insert_back(std::get<Ranks>(t)...);
            }
        };

        if constexpr (ReverseIndices && Extents::rank() >= 2) {
            // For CSC (reversed indices), we need to re-sort the COO
            // entries by column-major order before inserting into the
            // compressed structure, since COO compress() sorts row-major.
            //
            // Collect all entries, sort by reversed index order, then insert.
            struct Entry {
                std::array<index_type, Extents::rank()> indices;
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
                }(std::make_index_sequence<Extents::rank()>{});
                entry.value = v.value();
                entries.push_back(entry);
            }

            // Sort by reversed index order (col-major for rank-2)
            std::sort(entries.begin(), entries.end(),
                [](const Entry& a, const Entry& b) {
                    // Compare in reversed order: last index first
                    for (int d = Extents::rank() - 1; d >= 0; --d) {
                        if (a.indices[d] < b.indices[d]) return true;
                        if (a.indices[d] > b.indices[d]) return false;
                    }
                    return false;
                });

            for (const auto& entry : entries) {
                auto insert = [&R]<rank_type... Ranks>(
                    std::integer_sequence<rank_type, Ranks...>,
                    const std::array<index_type, Extents::rank()>& idx)
                    -> auto& {
                    return R.insert_back(
                        idx[Extents::rank() - 1 - Ranks]...);
                };
                insert(
                    std::make_integer_sequence<rank_type, Extents::rank()>{},
                    entry.indices) = entry.value;
            }
        } else {
            // CSR (natural order) or rank-1: iterate COO directly
            for (const auto& v : e) {
                m(std::make_integer_sequence<rank_type, Extents::rank()>{},
                  v.multiindex()) = v.value();
            }
        }
        return R;
    }
}

}  // namespace zipper::storage::detail
#endif
