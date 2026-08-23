#pragma once

#include <type_traits>

#include <zipper/types.hpp>

namespace zipper::utils::decomposition::detail {

template <typename T, index_type Axis>
inline constexpr index_type static_extent_v =
    std::decay_t<T>::extents_type::static_extent(Axis);

template <typename MatrixType>
concept StaticallySquare =
    static_extent_v<MatrixType, 0> == dynamic_extent ||
    static_extent_v<MatrixType, 1> == dynamic_extent ||
    static_extent_v<MatrixType, 0> == static_extent_v<MatrixType, 1>;

template <typename MatrixType, typename VectorType>
concept StaticallyCompatibleRhs =
    static_extent_v<MatrixType, 0> == dynamic_extent ||
    static_extent_v<VectorType, 0> == dynamic_extent ||
    static_extent_v<MatrixType, 0> == static_extent_v<VectorType, 0>;

template <typename MatrixType>
concept StaticallyNotWide =
    static_extent_v<MatrixType, 0> == dynamic_extent ||
    static_extent_v<MatrixType, 1> == dynamic_extent ||
    static_extent_v<MatrixType, 0> >= static_extent_v<MatrixType, 1>;

} // namespace zipper::utils::decomposition::detail
