#if !defined(ZIPPER_STORAGE_STLStorage_HPP)
#define ZIPPER_STORAGE_STLStorage_HPP
#include <array>
#include <cstdint>
#include <vector>

#include "zipper/types.hpp"

namespace zipper::storage {

namespace detail {
template <typename T>
struct StlStorageInfo {};
template <>
struct StlStorageInfo<std::vector<int64_t>> {
    constexpr static rank_type rank = 1;
    constexpr static index_type static_extent = std::dynamic_extent;
    static auto& get_value(std::vector<int64_t>& A, index_type i) {
        return A[i];
    }
    static const auto& get_value(const std::vector<int64_t>& A, index_type i) {
        return A[i];
    }
    // using extents_type = zipper::extents<std::dynamic_extent>();
    // static extents_type extents(const std::vector<int64_t>& A) {
    //     return extents_type(A.size());
    // }
};
template <size_t N>
struct StlStorageInfo<std::array<int64_t, N>> {
    constexpr static rank_type rank = 1;
    constexpr static index_type static_extent = index_type(N);
    static const auto& get_value(const std::array<int64_t, N>& A,
                                 index_type i) {
        return A[i];
    }
    static auto& get_value(std::array<int64_t, N>& A, index_type i) {
        return A[i];
    }

    // using extents_type = zipper::extents<N>();
    // constexpr static extents_type extents(const std::array<int64_t, N>& A) {
    //     return extents_type();
    // }
};
template <typename S>
struct StlStorageInfo<std::vector<S>> : public StlStorageInfo<S> {
    constexpr static rank_type rank = StlStorageInfo<S>::rank + 1;
    constexpr static index_type static_extent = std::dynamic_extent;
    template <typename... Args>
    static auto& get_value(std::vector<S>& A, index_type i, Args&&... args) {
        return StlStorageInfo<S>::get_value(A[i], std::forward<Args>(args)...);
    }
    template <typename... Args>
    static const auto& get_value(const std::vector<S>& A, index_type i,
                                 Args&&... args) {
        return StlStorageInfo<S>::get_value(A[i], std::forward<Args>(args)...);
    }
};
template <size_t N, typename S>
struct StlStorageInfo<std::array<S, N>> : public StlStorageInfo<S> {
    constexpr static rank_type rank = StlStorageInfo<S>::rank + 1;
    constexpr static index_type static_extent = index_type(N);
    template <typename... Args>
    static auto& get_value(std::array<S, N>& A, index_type i, Args&&... args) {
        return StlStorageInfo<S>::get_value(A[i], std::forward<Args>(args)...);
    }
    template <typename... Args>
    static const auto& get_value(const std::array<S, N>& A, index_type i,
                                 Args&&... args) {
        return StlStorageInfo<S>::get_value(A[i], std::forward<Args>(args)...);
    }
    // using extents_type = zipper::extents<N>();
    // constexpr static extents_type extents(const std::array<int64_t, N>& A) {
    //     return extents_type();
    // }
};

using extents_type =

}  // namespace detail

}  // namespace zipper::storage

#endif
