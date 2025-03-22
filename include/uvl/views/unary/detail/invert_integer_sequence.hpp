

#if !defined(UVL_VIEWS_UNARY_DETAIL_INVERT_INTEGER_SEQUENCE_HPP)
#define UVL_VIEWS_UNARY_DETAIL_INVERT_INTEGER_SEQUENCE_HPP
#include "uvl/types.hpp"

namespace uvl::views::unary::detail {

template <rank_type total_rank, rank_type... Indices>
struct invert_integer_sequence {
    template <rank_type... M>
    constexpr static auto make_arr(std::integer_sequence<rank_type, M...>) {
        std::array<rank_type, total_rank - sizeof...(Indices)> R;

        size_t index = 0;
        auto add = []<rank_type J>(std::integral_constant<rank_type, J>,
                                   auto& R, auto& index) {
            if (((J != Indices) && ...)) {
                R[index++] = J;
            }
        };

        (add(std::integral_constant<rank_type, M>{}, R, index), ...);
        return R;
    }
    constexpr static std::array<rank_type, total_rank - sizeof...(Indices)> R =
        make_arr(std::make_integer_sequence<rank_type, total_rank>{});

    template <rank_type... M>
    constexpr static auto make(std::integer_sequence<rank_type, M...>) {
        // return std::integer_sequence<rank_type, M...>{};
        return std::integer_sequence<rank_type, std::get<M>(R)...>{};
        // return std::integer_sequence<rank_type, std::get<M>(R)...>{};
    }

    constexpr static auto make() {
        return make(
            std::make_integer_sequence<rank_type,
                                       total_rank - sizeof...(Indices)>{});
    }

    using type = decltype(make());

    template <template <rank_type...>typename,typename>
        struct assign_types_i;

    template <template <rank_type...>typename rank_vartype,rank_type... types>
        struct assign_types_i<rank_vartype, std::integer_sequence<rank_type,types...>> {
            using type = rank_vartype<R[types]...>;
        };


    template <template <rank_type...> typename rank_vartype>
        using assign_types = assign_types_i<rank_vartype, decltype(std::make_integer_sequence<rank_type, total_rank - sizeof...(Indices)>{})>::type;
};

}  // namespace uvl::views::unary::detail
#endif
