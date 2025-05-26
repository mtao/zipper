
#if !defined(ZIPPER_STORAGE_CONCEPTS_DENSE_DATA_LIKE_HPP)
#define ZIPPER_STORAGE_CONCEPTS_DENSE_DATA_LIKE_HPP
#include <concepts>
#include <cstdlib>
#include <zipper/types.hpp>

#include "DataLike.hpp"

namespace zipper::storage::concepts {
template <typename T>
concept DenseDataLike = DataLike<T> && requires(T& t, const T& const_t) {
    {
        t.as_std_span()
    } -> std::same_as<std::span<typename T::value_type, T::static_size>>;
    {
        const_t.as_std_span()
    } -> std::same_as<std::span<typename T::value_type const, T::static_size>>;
};
}  // namespace zipper::storage::concepts

#endif
