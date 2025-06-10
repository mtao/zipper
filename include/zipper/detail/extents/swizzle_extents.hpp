#if !defined(ZIPPER_DETAIL_EXTENTS_SWIZZLE_EXTENTS_HPP)
#define ZIPPER_DETAIL_EXTENTS_SWIZZLE_EXTENTS_HPP

#include "dynamic_extents_indices.hpp"
#include "zipper/concepts/TupleLike.hpp"
#include "zipper/detail/ExtentsTraits.hpp"
#include "zipper/types.hpp"

namespace zipper::detail::extents {

template <index_type... SwizzleIndices>
struct ExtentsSwizzler {
    using swizzle_extents_type = zipper::extents<SwizzleIndices...>;
    using swizzle_extents_traits = ExtentsTraits<swizzle_extents_type>;

    constexpr static index_type size = sizeof...(SwizzleIndices);

    constexpr static rank_type valid_indices_rank =
        swizzle_extents_traits::rank_static;

    consteval static auto get_valid_indices()
        -> std::array<rank_type, valid_indices_rank> {
        rank_type r[valid_indices_rank];
        for (rank_type j = 0; j < swizzle_extents_type::rank(); ++j) {
            if (index_type index = swizzle_extents_type::static_extent(j);
                index != std::dynamic_extent) {
                r[index] = j;
            }
        }

        std::array<rank_type, valid_indices_rank> r2;
        for (rank_type j = 0; j < valid_indices_rank; ++j) {
            r2[j] = r[j];
        }

        return r2;
    }
    // swizzling can support dimensions that don't exist in the child view to
    // lift to higher order tensors (like making a vector a matrix) this keeps
    // track of which indices are ok or not
    constexpr static std::array<rank_type, valid_indices_rank>
        valid_internal_indices = get_valid_indices();

    // the required size of the object being swizzled
    constexpr static auto valid_internal_size = valid_internal_indices.size();

    consteval static auto get_indices()
        -> std::array<rank_type, valid_indices_rank> {
        std::array<rank_type, valid_indices_rank> r;
        for (rank_type j = 0; j < valid_indices_rank; ++j) {
            r[j] =
                swizzle_extents_type::static_extent(valid_internal_indices[j]);
        }
        return r;
    }

#if !defined(__cpp_pack_indexing)
    // this sort of tool isn't necessary with pack indexing
    constexpr static std::array<index_type, size> swizzle_indices = {
        {SwizzleIndices == std::dynamic_extent ? 1 : SwizzleIndices...}};
#endif
    template <index_type... Indices>
    using swizzled_extents_type =
        zipper::extents<SwizzleIndices == std::dynamic_extent
                            ? 1
                            : zipper::extents<Indices...>::static_extent(
                                  SwizzleIndices)...>;

    template <typename T>
    struct extents_type_swizzler {};
    template <index_type... Indices>
    struct extents_type_swizzler<zipper::extents<Indices...>> {
        using type = swizzled_extents_type<Indices...>;
    };
    template <typename T>
    using extents_type_swizzler_t = extents_type_swizzler<T>::type;

    // dim I,J,K with 0,2,1 is sent to I,K,J
    // say K is dynamic
    // then dynamic_indices is 1
    // dest_dyn_indices is 2

    template <index_type... Indices, std::size_t... N>
    static auto get_dynamic(const zipper::extents<Indices...>& e,
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
    static auto get_dynamic(const zipper::extents<Indices...>& e) {
        using my_extents = swizzled_extents_type<Indices...>;
        const auto sizes = get_dynamic(
            e, std::make_index_sequence<my_extents::rank_dynamic()>{});
        static_assert(my_extents::rank_dynamic() == sizes.size());
        return sizes;
    }

    template <index_type... Indices>
    static auto swizzle_extents(const zipper::extents<Indices...>& e) {
        using my_extents = swizzled_extents_type<Indices...>;
        if constexpr (my_extents::rank_dynamic() > 0) {
            const auto sizes = get_dynamic(e);
            static_assert(my_extents::rank_dynamic() == sizes.size());
            return my_extents(sizes);
        } else {
            return my_extents{};
        }
    }

    template <index_type Index>

        requires(Index < valid_internal_indices.size())
    constexpr static auto _unswizzle_single_index(
        concepts::TupleLike auto const& t) -> index_type {
        if constexpr (Index != std::dynamic_extent) {
            static_assert(valid_internal_indices[Index] < size);
            return std::get<valid_internal_indices[Index]>(t);
        } else {
            return 0;
        }
    }

    template <std::size_t... ValidIndices>
    constexpr static auto _unswizzle(concepts::TupleLike auto const& t,
                                     std::index_sequence<ValidIndices...>)
        -> std::array<index_type, valid_indices_rank> {
        using input_type = std::decay_t<decltype(t)>;
        constexpr std::size_t my_size = std::tuple_size_v<input_type>;
        static_assert(
            ((valid_internal_indices[ValidIndices] < my_size) && ...));
        static_assert(((SwizzleIndices < my_size ||
                        SwizzleIndices == std::dynamic_extent) &&
                       ...));
        std::array<index_type, valid_indices_rank> r{
            {_unswizzle_single_index<ValidIndices>(t)...}};
        return r;
    }

    constexpr static auto _unswizzle(concepts::TupleLike auto const& t)
        -> std::array<index_type, valid_indices_rank> {
        return _unswizzle(t, std::make_index_sequence<valid_indices_rank>{});
        // using input_type = std::decay_t<decltype(t)>;
        // constexpr std::size_t my_size = std::tuple_size_v<input_type>;
        // static_assert(((SwizzleIndices < my_size ||
        //                 SwizzleIndices == std::dynamic_extent) &&
        //                ...));
        // std::array<index_type, size> r{
        //     {_swizzle_single_index<SwizzleIndices>(t)...}};
        // return r;
    }

    template <typename... Indices>
    constexpr static auto unswizzle(Indices&&... indices)
        -> std::array<index_type, valid_indices_rank> {
        if constexpr (sizeof...(Indices) == 1 &&
                      (concepts::TupleLike<Indices> && ...)) {
            return _unswizzle(std::forward<Indices>(indices)...);
        } else {
#if defined(__cpp_pack_indexing) && false
            return {{index_type(indices...[SwizzledIndices`])...}};
#else
            // if we had parameter pack indexing this wouldn't be necsesary<
            // hoping the compiler inlines all of this
            return _unswizzle(std::make_tuple(indices...));
#endif
        }
    }
};

template <index_type... SwizzledIndices, index_type... Indices>
auto swizzle_extents(
    const zipper::extents<Indices...>& a,
    std::integer_sequence<index_type, SwizzledIndices...> = {}) {
    return ExtentsSwizzler<SwizzledIndices...>::swizzle_extents(a);
}

// template <index_type... Indices, index_type... SIndices>
// auto merge_extents(const zipper::extents<Indices...>& a,
// const zipper::extents<SIndices...>& b
//         ) {
//     return ExtentsSwizzler<zipper::extents<Indices...>, A>::extents(a);
// }
}  // namespace zipper::detail::extents
#endif
