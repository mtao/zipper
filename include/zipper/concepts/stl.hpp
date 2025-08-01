#if !defined(ZIPPER_CONCEPTS_STL_HPP)
#define ZIPPER_CONCEPTS_STL_HPP
#include <tuple>
#pragma GCC diagnostic push
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wdeprecated-literal-operator"
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <range/v3/utility/common_tuple.hpp>
#pragma GCC diagnostic pop


namespace zipper::concepts {
namespace detail {
    template<typename>
    struct is_stl_tuple : public std::false_type {};

    template<typename... Types>
    struct is_stl_tuple<std::tuple<Types...>> : public std::true_type {};
    template<typename... Types>
    struct is_stl_tuple<ranges::common_tuple<Types...>> : public std::true_type {};

    template<typename>
    struct is_stl_pair : public std::false_type {};
    template<typename A, typename B>
    struct is_stl_pair<std::pair<A, B>> : public std::true_type {};

    template<typename>
    struct is_stl_array : public std::false_type {};
    template<typename A, std::size_t N>
    struct is_stl_array<std::array<A, N>> : public std::true_type {};
}// namespace detail

template<typename T>
concept is_stl_tuple = detail::is_stl_tuple<std::decay_t<T>>::value;

template<typename T>
concept is_stl_array = detail::is_stl_array<std::decay_t<T>>::value;
}// namespace zipper::concepts
#endif
