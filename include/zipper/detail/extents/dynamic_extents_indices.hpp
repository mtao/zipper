#if !defined(ZIPPER_DETAIL_EXTENTS_DYNAMIC_EXTENT_INDICES_HPP)
#define ZIPPER_DETAIL_EXTENTS_DYNAMIC_EXTENT_INDICES_HPP

#include "zipper/types.hpp"

namespace zipper::detail::extents {

// template <index_type... Indices>
// constexpr auto dynamic_extent_indices(index_type... Indices) {
//     using E = extents<Indices...>;
// }
template <typename T>
struct DynamicExtentIndices;

template <index_type... indices>
struct DynamicExtentIndices<zipper::extents<indices...>> {
    using E = zipper::extents<indices...>;
    constexpr static rank_type rank_dynamic = E::rank_dynamic();
    consteval static auto _eval() {
        std::array<rank_type, rank_dynamic> r;
        rank_type idx = 0;
        for (rank_type j = 0; j < E::rank(); ++j) {
            if (E::static_extent(j) == std::dynamic_extent) {
                r[idx++] = j;
            }
        }
        return r;
    }
    constexpr static rank_type dynamic_rank = sizeof...(indices);
    using dynamic_indices_type = std::array<rank_type, sizeof...(indices)>;

    // composes the indices that belong to dynamic indices.
    // Note that due to swizzle the set of indices bieng
    template <std::size_t... Indices>
    consteval static const dynamic_indices_type get_dynamic_indices(
        std::index_sequence<Indices...>) {
        /*
        dynamic_indices_type ret;
        size_t index = 0;
        auto add = []<std::size_t J>(std::integral_constant<std::size_t, J>,
                                     auto& ret, size_t& index) {
            if (E::static_extent(J) == std::dynamic_extent) {
                ret[J] = index++;
            }
        };
        ((add(std::integral_constant<std::size_t, Indices>{}, ret, index),
          ...));

        return ret;
        */
        constexpr auto eval =
            []<rank_type J>(std::integral_constant<rank_type, J>) -> rank_type {
            for (rank_type j = 0; j < E::rank(); ++j) {
                if (E::static_extent(j) == J) {
                    return j;
                }
            }
            return -1;
        };
        return std::array<rank_type, dynamic_rank>{
            {eval(std::integral_constant<rank_type, Indices>{})...}};
    }

    static consteval dynamic_indices_type get_dynamic_indices() {
        return get_dynamic_indices(
            std::make_integer_sequence<rank_type, E::rank()>{});
    }

    constexpr static dynamic_indices_type dynamic_indices =
        get_dynamic_indices();
    constexpr static auto value = _eval();

    template <std::size_t... N>
    static auto run(const zipper::extents<indices...>& e,
                    std::integer_sequence<std::size_t, N...>) {
        return std::array<index_type, rank_dynamic>{
            {e.extent(std::get<N>(value))...}};
    }
    static auto run(const zipper::extents<indices...>& e) {
        return run(e, std::make_index_sequence<std::size_t(rank_dynamic)>{});
    }

    //
};

template <typename T>
constexpr auto dynamic_extents_indices_v = DynamicExtentIndices<T>::value;
template <typename T, typename U = T>
auto dynamic_extents(const T& extents) {
    return DynamicExtentIndices<U>::run(extents);
}

}  // namespace zipper::detail::extents
#endif
