#include <numeric>
#if !defined(ZIPPER_EXPRESSION_UNARY_DETAIL_INVERT_INTEGER_SEQUENCE_HPP)
#define ZIPPER_EXPRESSION_UNARY_DETAIL_INVERT_INTEGER_SEQUENCE_HPP
#include <tuple>

#include "zipper/detail/extents/dynamic_extents_indices.hpp"
#include "zipper/detail/extents/static_extents_to_array.hpp"
#include "zipper/types.hpp"

namespace zipper::expression::unary::detail {

// defines a sequence [a,b,c..f] by the entries in [0...total_rank] NOT
// identified in Indices
template <rank_type total_rank, rank_type... Indices>
struct invert_integer_sequence {
    consteval static bool in_sequence(rank_type J) {
        return ((J != Indices) && ...);
    }
    template <rank_type... M>
    consteval static auto make_reduced_to_full(
        std::integer_sequence<rank_type, M...>) {
        std::array<rank_type, total_rank - sizeof...(Indices)> R;

        size_t index = 0;
        auto add = []<rank_type J>(std::integral_constant<rank_type, J>,
                                   auto& R_, auto& index_) {
            if (in_sequence(J)) {
                R_[index_++] = J;
            }
        };

        (add(std::integral_constant<rank_type, M>{}, R, index), ...);
        return R;
    }

    template <rank_type... M>
    consteval static auto make_paired_indices(
        std::integer_sequence<rank_type, M...>) {
        constexpr std::array<rank_type, sizeof...(Indices)> Inds = {{Indices...}};
        constexpr size_t size = sizeof...(Indices);

        std::array<rank_type, sizeof...(M)> A = {{Inds[M]...}};
        std::array<rank_type, sizeof...(M)> B = {{Inds[size - M]...}};

        return std::array{{A,B}};
    }
    constexpr static std::array<std::array<rank_type, sizeof...(Indices)>,2> paired_indices = make_paired_indices(
            std::make_integer_sequence<rank_type, sizeof...(Indices) / 2>{}
            );

    // makes an array mapping each reduced index
    constexpr static std::array<rank_type, total_rank - sizeof...(Indices)>
        reduced_rank_to_full_indices = make_reduced_to_full(
            std::make_integer_sequence<rank_type, total_rank>{});

    template <rank_type... M>
    consteval static auto make_full_to_reduced(
        std::integer_sequence<rank_type, M...>) {
        constexpr auto eval = []<rank_type J>(std::integral_constant<rank_type, J>) -> rank_type {
            for(size_t j = 0; j < reduced_rank_to_full_indices.size(); ++j) {
                if(reduced_rank_to_full_indices[j] == J) {
                    return j;
                }
            }
            return -1;
        };
        return std::array<rank_type, total_rank>{{eval(std::integral_constant<rank_type, M>{})...}};
    }

    constexpr static std::array<rank_type, total_rank>
        full_rank_to_reduced_indices = make_full_to_reduced(
            std::make_integer_sequence<rank_type,
                                       total_rank>{});

    template <rank_type... M>
    consteval static auto make(std::integer_sequence<rank_type, M...>) {
        // return std::integer_sequence<rank_type, M...>{};
        return std::integer_sequence<
            rank_type, std::get<M>(reduced_rank_to_full_indices)...>{};
        // return std::integer_sequence<rank_type, std::get<M>(R)...>{};
    }

    consteval static auto make() {
        return make(
            std::make_integer_sequence<rank_type,
                                       total_rank - sizeof...(Indices)>{});
    }

    using type = decltype(make());

    template <template <rank_type...> typename assigned_type,
              std::array<index_type, total_rank>,
              typename IterationIntegerSequence>
    struct assign_types_i;

    template <template <index_type...> typename rank_vartype,
              std::array<index_type, total_rank> Array, rank_type... N>
    struct assign_types_i<rank_vartype, Array,
                          std::integer_sequence<rank_type, N...>> {
        using type = rank_vartype<Array[reduced_rank_to_full_indices[N]]...>;
    };

    template <template <rank_type...> typename rank_vartype,
              std::array<index_type, total_rank> Arr>
    using assign_types = assign_types_i<
        rank_vartype, Arr,
        decltype(std::make_integer_sequence<
                 rank_type, total_rank - sizeof...(Indices)>{})>::type;

    template <rank_type... N, std::size_t... DN>
    constexpr static auto get_extents(
        const auto& o, std::integer_sequence<rank_type, N...>,
        std::integer_sequence<std::size_t, DN...>) {
        using extents_type =
            assign_types<zipper::extents,
                         zipper::detail::extents::static_extents_to_array_v<
                             std::decay_t<decltype(o)>>>;

        constexpr auto& dyn_inds =
            zipper::detail::extents::dynamic_extents_indices_v<extents_type>;
        return extents_type(std::array<index_type, sizeof...(DN)>{
            {o.extent(reduced_rank_to_full_indices[dyn_inds[DN]])...}});
        // return extents_type<
        //>(
        //  );
    }

    constexpr static auto get_extents(const auto& o) {
        using extents_type =
            assign_types<zipper::extents,
                         zipper::detail::extents::static_extents_to_array_v<
                             std::decay_t<decltype(o)>>>;
        using dyn_exts_type =
            std::decay_t<decltype(zipper::detail::extents::
                                      dynamic_extents_indices_v<extents_type>)>;
        constexpr std::size_t dyn_size = std::tuple_size<dyn_exts_type>::value;
        return get_extents(
            o,
            std::make_integer_sequence<rank_type,
                                       total_rank - sizeof...(Indices)>{}

            ,
            std::make_integer_sequence<std::size_t, dyn_size>{});
    }
};

}  // namespace zipper::expression::unary::detail
#endif
