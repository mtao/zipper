#pragma once
#include "assignment_source.hpp"

namespace zipper {

template <typename T, size_t N>
    requires detail::fixed_numeric_array<std::array<T, N>>::value
auto assignment_source_traits<std::array<T, N>>::get(const source_type &source)
    -> expression_type {
    return expression_type(source);
}

template <typename T, typename Allocator>
    requires(std::is_arithmetic_v<T> && !std::is_same_v<T, bool> &&
             !std::is_volatile_v<T>)
auto assignment_source_traits<std::vector<T, Allocator>>::get(const source_type &source)
    -> expression_type {
    return expression_type(std::span<const T>(source.data(), source.size()));
}

template <typename T>
    requires detail::fixed_numeric_array<T>::value
auto assignment_source_traits<std::vector<T>>::get(const source_type &source)
    -> expression_type {
    return expression_type(source);
}

template <typename T, size_t N>
    requires(std::is_arithmetic_v<std::remove_const_t<T>> &&
             !std::is_volatile_v<T>)
auto assignment_source_traits<std::span<T, N>>::get(const source_type &source)
    -> expression_type {
    return expression_type(std::span<const std::remove_const_t<T>, N>(source));
}

template <typename T, typename Extents, typename Layout, typename Accessor>
    requires(std::is_arithmetic_v<std::remove_const_t<T>> &&
             !std::is_volatile_v<T> && Extents::rank() > 0)
auto assignment_source_traits<mdspan<T, Extents, Layout, Accessor>>::get(
    const source_type &source) -> expression_type {
    return expression_type(source);
}

} // namespace zipper
