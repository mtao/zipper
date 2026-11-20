#pragma once

#include "expression/nullary/MDSpan.hpp"
#include "expression/nullary/StlMDArray.hpp"

#include <array>
#include <span>
#include <type_traits>
#include <vector>

namespace zipper {

/// Opt-in adaptation of non-zipper assignment sources. The primary template
/// deliberately provides no adapter. Specialize for a user-defined type and
/// provide `get(const T&)`, returning a zipper wrapper or raw expression by
/// value. Its referenced data need only remain valid until assignment returns;
/// assignment never retains the adapter. Do not return a dangling expression.
template <typename T>
struct assignment_source_traits {};

namespace detail {

template <typename T>
struct fixed_numeric_array : std::false_type {};

template <typename T, size_t N>
    requires(std::is_arithmetic_v<T> && !std::is_volatile_v<T>)
struct fixed_numeric_array<std::array<T, N>> : std::true_type {};

template <typename T, size_t N>
    requires fixed_numeric_array<T>::value
struct fixed_numeric_array<std::array<T, N>> : std::true_type {};

} // namespace detail

/// Fixed, rectangular std::array compositions with arithmetic leaves.
template <typename T, size_t N>
    requires detail::fixed_numeric_array<std::array<T, N>>::value
struct assignment_source_traits<std::array<T, N>> {
    using source_type = std::array<T, N>;
    using expression_type = expression::nullary::StlMDArray<const source_type &>;
    static auto get(const source_type &source) -> expression_type;
};

/// Flat vectors, including custom allocators; packed vector<bool> is excluded.
template <typename T, typename Allocator>
    requires(std::is_arithmetic_v<T> && !std::is_same_v<T, bool> &&
             !std::is_volatile_v<T>)
struct assignment_source_traits<std::vector<T, Allocator>> {
    using source_type = std::vector<T, Allocator>;
    using expression_type = expression::nullary::MDSpan<const T, dextents<1>>;
    static auto get(const source_type &source) -> expression_type;
};

/// Vectors of fixed numeric arrays (default allocator), never ragged vectors.
template <typename T>
    requires detail::fixed_numeric_array<T>::value
struct assignment_source_traits<std::vector<T>> {
    using source_type = std::vector<T>;
    using expression_type = expression::nullary::StlMDArray<const source_type &>;
    static auto get(const source_type &source) -> expression_type;
};

template <typename T, size_t N>
    requires(std::is_arithmetic_v<std::remove_const_t<T>> &&
             !std::is_volatile_v<T>)
struct assignment_source_traits<std::span<T, N>> {
    using source_type = std::span<T, N>;
    using expression_type = expression::nullary::MDSpan<
        const std::remove_const_t<T>, extents<N>>;
    static auto get(const source_type &source) -> expression_type;
};

template <typename T, typename Extents, typename Layout, typename Accessor>
    requires(std::is_arithmetic_v<std::remove_const_t<T>> &&
             !std::is_volatile_v<T> && Extents::rank() > 0)
struct assignment_source_traits<mdspan<T, Extents, Layout, Accessor>> {
    using source_type = mdspan<T, Extents, Layout, Accessor>;
    using expression_type = expression::nullary::MDSpan<T, Extents, Layout, Accessor>;
    static auto get(const source_type &source) -> expression_type;
};

} // namespace zipper

#include "assignment_source.hxx"
