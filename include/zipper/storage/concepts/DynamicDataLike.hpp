#if !defined(ZIPPER_STORAGE_CONCEPTS_DYNAMIC_DATA_LIKE_HPP)
#define ZIPPER_STORAGE_CONCEPTS_DYNAMIC_DATA_LIKE_HPP
#include <concepts>
#include <cstdlib>
#include <zipper/types.hpp>

#include "DataLike.hpp"

namespace zipper::storage::concepts {
template <typename T>
concept DynamicDataLike =
    DataLike<T> && T::static_size == std::dynamic_extent &&
    requires(T& t, index_type index) {
        { t.resize(index) };
    };
}  // namespace zipper::storage::concepts

#endif
