
#if !defined(UVL_DETAIL_DYNAMIC_EXTENT_INDICES_HPP)
#define UVL_DETAIL_DYNAMIC_EXTENT_INDICES_HPP

#include "uvl/types.hpp"

namespace uvl::detail {

// template <index_type... Indices>
// constexpr auto dynamic_extent_indices(index_type... Indices) {
//     using E = extents<Indices...>;
// }
template <typename T>
struct DynamicExtentIndices;

template <index_type... indices>
struct DynamicExtentIndices<extents<indices...>> {
    using E = extents<indices...>;
    constexpr static rank_type rank_dynamic = E::rank_dynamic();
    constexpr static auto _eval() {
        std::array<rank_type, rank_dynamic> r;
        rank_type idx = 0;
        for (rank_type j = 0; j < E::rank(); ++j) {
            if (E::static_extent(j) == std::dynamic_extent) {
                r[idx++] = j;
            }
        }
        return r;
    }

    constexpr static auto value = _eval();

    template <std::size_t... N>
    static auto run(const extents<indices...>& e,
                    std::integer_sequence<std::size_t, N...>) {
        return std::array<index_type, rank_dynamic>{
            {e.extent(std::get<N>(value))...}};
    }
    static auto run(const extents<indices...>& e) {
        return run(e, std::make_index_sequence<std::size_t(rank_dynamic)>{});
    }

    //
};

template <typename T>
constexpr static auto dynamic_extents_indices_v =
    DynamicExtentIndices<T>::value;
template <typename T, typename U = T>
auto dynamic_extents(const T& extents) {
    return DynamicExtentIndices<U>::run(extents);
}

}  // namespace uvl::detail
#endif
