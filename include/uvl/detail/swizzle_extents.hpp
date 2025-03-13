#if !defined(UVL_DETAIL_SWIZZLE_EXTENTS_HPP)
#define UVL_DETAIL_SWIZZLE_EXTENTS_HPP

#include "dynamic_extents_indices.hpp"
#include "uvl/concepts/TupleLike.hpp"
#include "uvl/types.hpp"

namespace uvl::detail {

template <index_type... SwizzleIndices>
struct ExtentsSwizzler {
    using swizzle_extents_type = extents<SwizzleIndices...>;

    constexpr static index_type size = sizeof...(SwizzleIndices);
#if !defined(__cpp_pack_indexing)
    // this sort of tool isn't necessary with pack indexing
    constexpr static std::array<index_type, size> swizzle_indices = {
        {SwizzleIndices...}};
#endif
    template <index_type... Indices>
    using swizzled_extents_type =
        extents<extents<Indices...>::static_extent(SwizzleIndices)...>;

    template <typename T>
    struct extents_type_swizzler {};
    template <index_type... Indices>
    struct extents_type_swizzler<extents<Indices...>> {
        using type = swizzled_extents_type<Indices...>;
    };
    template <typename T>
    using extents_type_swizzler_t = extents_type_swizzler<T>::type;

    // dim I,J,K with 0,2,1 is sent to I,K,J
    // say K is dynamic
    // then dynamic_indices is 1
    // dest_dyn_indices is 2

    template <index_type... Indices, std::size_t... N>
    static auto get_dynamic(const extents<Indices...>& e,
                            std::integer_sequence<std::size_t, N...>) {
        using my_extents = swizzled_extents_type<Indices...>;
        constexpr static auto dynamic_indices =
            dynamic_extents_indices_v<my_extents>;

        static_assert(dynamic_indices.size() == my_extents::rank_dynamic());

        constexpr static rank_type rank_dynamic = my_extents::rank_dynamic();

#if defined(__cpp_pack_indexing)
        constexpr static std::array<index_type, rank_dynamic>
            dynamic_swizzled_indices{
                {SwizzleIndices...[dynamic_indices[N]]...}};
#else
        constexpr static std::array<index_type, rank_dynamic>
            dynamic_swizzled_indices{
                {std::get<std::get<N>(dynamic_indices)>(swizzle_indices)...}};
#endif

        return std::array<index_type, rank_dynamic>{
            {e.extent(std::get<N>(dynamic_swizzled_indices))...}};
    }
    template <index_type... Indices>
    static auto get_dynamic(const extents<Indices...>& e) {
        using my_extents = swizzled_extents_type<Indices...>;
        const auto sizes = get_dynamic(
            e, std::make_index_sequence<my_extents::rank_dynamic()>{});
        static_assert(my_extents::rank_dynamic() == sizes.size());
        return sizes;
    }

    template <index_type... Indices>
    static auto swizzle_extents(const extents<Indices...>& e) {
        using my_extents = swizzled_extents_type<Indices...>;
        if constexpr (my_extents::rank_dynamic() > 0) {
            const auto sizes = get_dynamic(e);
            static_assert(my_extents::rank_dynamic() == sizes.size());
            return my_extents(sizes);
        } else {
            return my_extents{};
        }
    }

    constexpr static auto _swizzle(concepts::TupleLike auto const& t)
        -> std::array<index_type, size> {
        using input_type = std::decay_t<decltype(t)>;
        constexpr std::size_t my_size = std::tuple_size_v<input_type>;
        static_assert(((SwizzleIndices < my_size) && ...));
        std::array<index_type, size> r{
            {index_type(std::get<SwizzleIndices>(t))...}};
        return r;
    }

    template <typename... Indices>
    constexpr static auto swizzle(Indices&&... indices)
        -> std::array<index_type, size> {
        if constexpr (sizeof...(Indices) == 1 &&
                      (concepts::TupleLike<Indices> && ...)) {
            return _swizzle(std::forward<Indices>(indices)...);
        } else {
#if defined(__cpp_pack_indexing) && false
            return {{index_type(indices...[SwizzledIndices])...}};
#else
            // if we had parameter pack indexing this wouldn't be necsesary<
            // hoping the compiler inlines all of this
            return _swizzle(std::make_tuple(indices...));
#endif
        }
    }
};

template <index_type... SwizzledIndices, index_type... Indices>
auto swizzle_extents(const extents<Indices...>& a,
                     std::integer_sequence<index_type, SwizzledIndices...>) {
    return ExtentsSwizzler<SwizzledIndices...>::swizzle_extents(a);
}

// template <index_type... Indices, index_type... SIndices>
// auto merge_extents(const extents<Indices...>& a,
// const extents<SIndices...>& b
//         ) {
//     return ExtentsSwizzler<extents<Indices...>, A>::extents(a);
// }
}  // namespace uvl::detail
#endif
