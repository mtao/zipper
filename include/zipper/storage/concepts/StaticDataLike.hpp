#if !defined(ZIPPER_STORAGE_CONCEPTS_STATIC_DATA_LIKE_HPP)
#define ZIPPER_STORAGE_CONCEPTS_STATIC_DATA_LIKE_HPP
#include <concepts>
#include <cstdlib>
#include <zipper/types.hpp>

#include "DataLike.hpp"

namespace zipper::storage::concepts {
template <index_type N, typename T>
concept StaticDataLike = DataLike<T> && T::static_size == N;
}  // namespace zipper::storage::concepts

#endif
