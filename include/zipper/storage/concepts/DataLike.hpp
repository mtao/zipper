#if !defined(ZIPPER_STORAGE_CONCEPTS_DATA_LIKE_HPP)
#define ZIPPER_STORAGE_CONCEPTS_DATA_LIKE_HPP
#include <concepts>
#include <cstdlib>
#include <zipper/types.hpp>

namespace zipper::storage::concepts {
template <typename T>
concept DataLike = requires(T &t) {
    // mutable iterators
    { t.begin() } -> std::same_as<typename T::iterator_type>;
    { t.end() } -> std::same_as<typename T::iterator_type>;
} && requires(const T &t) {
    // size of the data
    { t.size() } -> std::same_as<std::size_t>;

    // const iterator types
    { t.begin() } -> std::same_as<typename T::const_iterator_type>;
    { t.end() } -> std::same_as<typename T::const_iterator_type>;
    { t.cbegin() } -> std::same_as<typename T::const_iterator_type>;
    { t.cend() } -> std::same_as<typename T::const_iterator_type>;
} && requires(T &t, index_type index) {
    // const reference access
    { t.coeff_ref(index) } -> std::same_as<typename T::value_type &>;
} && requires(const T &t, index_type index) {
    // coefficient by value
    { t.coeff(index) } -> std::same_as<typename T::value_type>;
    // const reference access
    {
        t.const_coeff_ref(index)
    } -> std::same_as<const typename T::value_type &>;
};
}  // namespace zipper::storage::concepts

#endif
