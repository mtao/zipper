#if !defined(ZIPPER_STORAGE_CONCEPTS_DATA_LIKE_HPP)
#define ZIPPER_STORAGE_CONCEPTS_DATA_LIKE_HPP
#include <concepts>
#include <cstdlib>
#include <zipper/types.hpp>

namespace zipper::storage::concepts {
template <typename T>
concept DataLike = requires(T &t, const T &const_t, const T::value_type &value,
                            index_type index) {
    { const_t.size() } -> std::same_as<std::size_t>;
    { const_t.coeff(index) } -> std::same_as<typename T::value_type>;
    { t.coeff_ref(index) } -> std::same_as<typename T::value_type &>;
    {
        const_t.const_coeff_ref(index)
    } -> std::same_as<const typename T::value_type &>;
    { const_t.begin() } -> std::same_as<typename T::value_type const *>;
    { const_t.end() } -> std::same_as<typename T::value_type const *>;

    { t.begin() } -> std::same_as<typename T::value_type *>;
    { t.end() } -> std::same_as<typename T::value_type *>;
    { t.cbegin() } -> std::same_as<typename T::value_type const *>;
    { t.cend() } -> std::same_as<typename T::value_type const *>;
};
}  // namespace zipper::storage::concepts

#endif
