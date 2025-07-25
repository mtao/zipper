#if !defined(ZIPPER_STORAGE_DETAIL_SPARSE_LAYOUT_POLICY_HPP)
#define ZIPPER_STORAGE_DETAIL_SPARSE_LAYOUT_POLICY_HPP

#include <cstdint>
#include <map>
#include <vector>
#include <zipper/types.hpp>

namespace zipper::storage::detail {

enum class SparseLayout : int8_t { Full = 0, Sparse = 1 };
template <SparseLayout... Layouts>
struct SparseLayoutPolicy {};

template <>
struct SparseLayoutPolicy<SparseLayout::Full> {};

template <>
struct SparseLayoutPolicy<SparseLayout::Sparse> {
    std::map<index_type, index_type> _data;

    index_type operator()(index_type index) const { return }
};

template <SparseLayout... Rest>
struct SparseLayoutPolicy<SparseLayout::Full, Rest...> {};
template <SparseLayout... Rest>
struct SparseLayoutPolicy<SparseLayout::Sparse, Rest...> {};

#endif
