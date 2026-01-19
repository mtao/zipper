
#if !defined(ZIPPER_STORAGE_CONCEPTS_DENSE_DATA_LIKE_HPP)
#define ZIPPER_STORAGE_CONCEPTS_DENSE_DATA_LIKE_HPP
#include <concepts>
#include <zipper/types.hpp>

#include "Data.hpp"

namespace zipper::storage::concepts {
template <typename T>
concept DenseData = Data<T> && requires(T &t) {
  // can create a mutable span
  {
    t.as_std_span()
  } -> std::same_as<std::span<typename T::value_type, T::static_size>>;
} && requires(const T &t) {
  {
    // can create a const span
    t.as_std_span()
  } -> std::same_as<std::span<typename T::value_type const, T::static_size>>;
};
} // namespace zipper::storage::concepts

#endif
