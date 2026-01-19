#if !defined(ZIPPER_STORAGE_CONCEPTS_STATICDATA_HPP)
#define ZIPPER_STORAGE_CONCEPTS_STATICDATA_HPP
#include <zipper/types.hpp>

#include "Data.hpp"

namespace zipper::storage::concepts {
template <index_type N, typename T>
concept StaticData = Data<T> && T::static_size == N;
} // namespace zipper::storage::concepts

#endif
