

#if !defined(UVL_VIEWS_UNARY_DETAIL_INVERT_INTEGER_SEQUENCE_HPP)
#define UVL_VIEWS_UNARY_DETAIL_INVERT_INTEGER_SEQUENCE_HPP
#include "uvl/types.hpp"

namespace uvl::views::unary::detail {

// defines a sequence [a,b,c..f] by the entries in [0...total_rank] NOT
// identified in Indices
template <rank_type total_rank, rank_type... Indices>
struct invert_integer_sequence {
    constexpr static bool in_sequence(rank_type J) {
        return ((J != Indices) && ...);
    }
    template <rank_type... M>
    constexpr static auto make_reduced_to_full(
        std::integer_sequence<rank_type, M...>) {
        std::array<rank_type, total_rank - sizeof...(Indices)> R;

        size_t index = 0;
        auto add = []<rank_type J>(std::integral_constant<rank_type, J>,
                                   auto& R, auto& index) {
            if (in_sequence(J)) {
                R[index++] = J;
            }
        };

        (add(std::integral_constant<rank_type, M>{}, R, index), ...);
        return R;
    }

    // makes an array mapping each reduced index
    constexpr static std::array<rank_type, total_rank - sizeof...(Indices)>
        reduced_rank_to_full_indices = make_reduced_to_full(
            std::make_integer_sequence<rank_type, total_rank>{});

    template <rank_type... M>
    constexpr static auto make_full_to_reduced(
        std::integer_sequence<rank_type, M...>) {
        std::array<rank_type, total_rank> R;
        auto add = []<rank_type J>(std::integral_constant<rank_type, J>,
                                   auto& R) {
            R[reduced_rank_to_full_indices[J]] = J;
        };

        (add(std::integral_constant<rank_type, M>{}, R), ...);
        return R;
    }

    constexpr static std::array<rank_type, total_rank>
        full_rank_to_reduced_indices = make_full_to_reduced(
            std::make_integer_sequence<rank_type,
                                       total_rank - sizeof...(Indices)>{});

    template <rank_type... M>
    constexpr static auto make(std::integer_sequence<rank_type, M...>) {
        // return std::integer_sequence<rank_type, M...>{};
        return std::integer_sequence<
            rank_type, std::get<M>(reduced_rank_to_full_indices)...>{};
        // return std::integer_sequence<rank_type, std::get<M>(R)...>{};
    }

    constexpr static auto make() {
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
                          std::integer_sequence<rank_type, N...> > {
        using type = rank_vartype<Array[reduced_rank_to_full_indices[N]]...>;
    };

    template <template <rank_type...> typename rank_vartype,
              std::array<index_type, total_rank> Arr>
    using assign_types = assign_types_i<
        rank_vartype, Arr,
        decltype(std::make_integer_sequence<
                 rank_type, total_rank - sizeof...(Indices)>{})>::type;
};

}  // namespace uvl::views::unary::detail
#endif
