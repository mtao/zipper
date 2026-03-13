#if !defined(ZIPPER_CONCEPTS_STL_HPP)
#define ZIPPER_CONCEPTS_STL_HPP
#include <tuple>
#include <array>
#include <vector>
#include <cstddef>
#include <type_traits>


namespace zipper::concepts {
namespace detail {
    template<typename>
    struct is_stl_tuple : public std::false_type {};

    template<typename... Types>
    struct is_stl_tuple<std::tuple<Types...>> : public std::true_type {};

    template<typename>
    struct is_stl_pair : public std::false_type {};
    template<typename A, typename B>
    struct is_stl_pair<std::pair<A, B>> : public std::true_type {};

    template<typename>
    struct is_stl_array : public std::false_type {};
    template<typename A, std::size_t N>
    struct is_stl_array<std::array<A, N>> : public std::true_type {};

    /// Detects compositions of std::vector/std::array that StlStorageInfo
    /// can handle.  The leaf element type must be arithmetic.
    /// Examples:
    ///   std::array<double, 3>                      → rank 1
    ///   std::vector<int>                            → rank 1
    ///   std::vector<std::array<double, 3>>          → rank 2
    ///   std::array<std::array<double, 3>, 4>        → rank 2
    template<typename T>
    struct is_stl_storage : public std::false_type {};

    // std::array<S, N> where S is arithmetic (rank-1 leaf)
    template<typename S, std::size_t N>
        requires(std::is_arithmetic_v<S>)
    struct is_stl_storage<std::array<S, N>> : public std::true_type {};

    // std::array<S, N> where S is itself an STL storage (rank > 1)
    template<typename S, std::size_t N>
        requires(is_stl_storage<S>::value)
    struct is_stl_storage<std::array<S, N>> : public std::true_type {};

    // std::vector<S> where S is arithmetic (rank-1 leaf)
    template<typename S>
        requires(std::is_arithmetic_v<S>)
    struct is_stl_storage<std::vector<S>> : public std::true_type {};

    // std::vector<S> where S is itself an STL storage (rank > 1)
    template<typename S>
        requires(is_stl_storage<S>::value)
    struct is_stl_storage<std::vector<S>> : public std::true_type {};

    /// Computes the rank of an STL storage composition.
    template<typename T>
    struct stl_storage_rank;

    template<typename S, std::size_t N>
    struct stl_storage_rank<std::array<S, N>> {
        static constexpr std::size_t value = 1 + stl_storage_rank<S>::value;
    };
    template<typename S>
    struct stl_storage_rank<std::vector<S>> {
        static constexpr std::size_t value = 1 + stl_storage_rank<S>::value;
    };
    // Leaf (arithmetic type)
    template<typename S>
        requires(std::is_arithmetic_v<S>)
    struct stl_storage_rank<S> {
        static constexpr std::size_t value = 0;
    };

}// namespace detail

template<typename T>
concept is_stl_tuple = detail::is_stl_tuple<std::decay_t<T>>::value;

template<typename T>
concept is_stl_array = detail::is_stl_array<std::decay_t<T>>::value;

/// True for std::vector/std::array compositions that StlStorageInfo handles.
template<typename T>
concept StlStorage = detail::is_stl_storage<std::decay_t<T>>::value;

/// True for StlStorage types with a specific rank.
template<typename T, std::size_t R>
concept StlStorageOfRank = StlStorage<T> &&
    (detail::stl_storage_rank<std::decay_t<T>>::value == R);

}// namespace zipper::concepts
#endif
