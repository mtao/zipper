#if !defined(ZIPPER_STORAGE_CONCEPTS_DYNAMICDATA_HPP)
#define ZIPPER_STORAGE_CONCEPTS_DYNAMICDATA_HPP
#include <zipper/types.hpp>

#include "Data.hpp"

namespace zipper::storage::concepts {
template <typename T>
concept DynamicData = Data<T> && T::static_size == std::dynamic_extent &&
                      requires(T &t, index_type index) {
                        { t.resize(index) };
                      };
} // namespace zipper::storage::concepts

#endif
