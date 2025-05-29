#if !defined(ZIPPER_STORAGE_DETAIL_TO_SPARSE_COMPRESSED_DATA_HPP)
#define ZIPPER_STORAGE_DETAIL_TO_SPARSE_COMPRESSED_DATA_HPP
#include <spdlog/spdlog.h>

#include "SparseCompressedData.hpp"
#include "zipper/storage/SparseCoordinateAccessor.hpp"

namespace zipper::storage::detail {
template <bool ReverseIndices = false, typename ValueType = double,
          typename Extents = dextents<0>()>
SparseCompressedData<ValueType, Extents::rank() - 1> to_sparse_compressed_data(
    const SparseCoordinateAccessor<ValueType, Extents>& e) {
    if (!e.is_compressed()) {
        spdlog::debug(
            "Converting SparseCoordinateAccessor to SpasreCompressedData "
            "without compressing the sparse coordinate data first, creating a "
            "copy to compress internally");
        SparseCoordinateAccessor<ValueType, Extents> e2 = e;
        e2.compress();
        return to_sparse_compressed_data(e2);
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

        for (const auto& v : e) {
            m(std::make_integer_sequence<rank_type, Extents::rank()>{},
              v.multiindex()) = v.value();
        }
        return R;
    }
}

}  // namespace zipper::storage::detail
#endif
