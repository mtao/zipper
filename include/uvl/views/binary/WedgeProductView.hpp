#if !defined(UVL_VIEWS_BINARY_WEDGEPRODUCTVIEW_HPP)
#define UVL_VIEWS_BINARY_WEDGEPRODUCTVIEW_HPP

#include "BinaryViewBase.hpp"
#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/detail/pack_index.hpp"

namespace uvl::views {
namespace binary {
template <concepts::ViewDerived A, concepts::ViewDerived B>
class WedgeProductView;

}
namespace detail {

template <typename A, typename B>
struct wedge_coeffwise_extents_values;
template <index_type... A, index_type... B>
struct wedge_coeffwise_extents_values<extents<A...>, extents<B...>> {
    using product_extents_type = uvl::extents<A..., B...>;
    using a_extents_type = extents<A...>;
    using b_extents_type = extents<B...>;

    static_assert(product_extents_type::rank() ==
                  a_extents_type::rank() + b_extents_type::rank());

    using a_extents_traits = uvl::detail::ExtentsTraits<a_extents_type>;
    using b_extents_traits = uvl::detail::ExtentsTraits<b_extents_type>;

    template <rank_type... N, rank_type... M>
    constexpr static product_extents_type merge(
        const extents<A...>& a, const extents<B...>& b,
        std::integer_sequence<rank_type, N...>,
        std::integer_sequence<rank_type, M...>) {
        using AEI = a_extents_traits::dynamic_indices_helper;
        using BEI = b_extents_traits::dynamic_indices_helper;

        constexpr auto ADI = AEI::dynamic_indices;
        constexpr auto BDI = BEI::dynamic_indices;

        return product_extents_type(a.extent(ADI[N])..., b.extent(BDI[M])...);
    }

    constexpr static product_extents_type merge(const extents<A...>& a,
                                                const extents<B...>& b) {
        return merge(
            a, b,
            std::make_integer_sequence<rank_type, rank_type(sizeof...(A))>{},
            std::make_integer_sequence<rank_type, rank_type(sizeof...(B))>{});
    }
};

}  // namespace detail
template <typename A, typename B>
struct detail::ViewTraits<binary::WedgeProductView<A, B>>
    : public binary::detail::DefaultBinaryViewTraits<A, B>
{
    using ATraits = views::detail::ViewTraits<A>;
    constexpr static rank_type lhs_rank = ATraits::extents_type::rank();
    using BTraits = views::detail::ViewTraits<B>;
    constexpr static rank_type rhs_rank = BTraits::extents_type::rank();
    using CEV =
        detail::wedge_coeffwise_extents_values<typename ATraits::extents_type,
                                                typename BTraits::extents_type>;

    static_assert(std::is_same_v<typename CEV::a_extents_type,
                                 typename ATraits::extents_type>);
    static_assert(std::is_same_v<typename CEV::b_extents_type,
                                 typename BTraits::extents_type>);
    using extents_type = CEV::product_extents_type;
    static_assert(
        std::is_same_v<typename CEV::product_extents_type, extents_type>);
    static_assert(extents_type::rank() == lhs_rank + rhs_rank);

    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
};

namespace binary {
template <concepts::ViewDerived A, concepts::ViewDerived B>
class WedgeProductView : public BinaryViewBase<WedgeProductView<A, B>, A, B> {
   public:
    using self_type = WedgeProductView<A, B>;
    using ViewBase<self_type>::operator();
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using value_type = traits::value_type;
    using Base = BinaryViewBase<self_type, A, B>;
    using Base::extent;
    using Base::lhs;
    using Base::rhs;
    constexpr static rank_type lhs_rank = traits::lhs_rank;
    constexpr static rank_type rhs_rank = traits::rhs_rank;

    using extents_type = traits::extents_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;

    WedgeProductView(const A& a, const B& b)
        requires(extents_traits::is_static)
        : Base(a, b) {
    }
    WedgeProductView(const A& a, const B& b)
        requires(!extents_traits::is_static)
        : Base(a, b, traits::CEV::merge(a.extents(), b.extents())) {}

    template <bool DoOffset, typename... Args, rank_type... ranks>
    auto lhs_value(std::integer_sequence<rank_type, ranks...>,
                   Args&&... args) const -> decltype(auto) {
        return lhs()(
            uvl::detail::pack_index<ranks + (DoOffset ? rhs_rank : 0)>(std::forward<Args>(args)...)...);
    }
    template <bool DoOffset, typename... Args, rank_type... ranks>
    auto rhs_value(std::integer_sequence<rank_type, ranks...>,
                   Args&&... args) const -> decltype(auto) {
        return rhs()(uvl::detail::pack_index<ranks + (DoOffset ? lhs_rank : 0)>(
            std::forward<Args>(args)...)...);
    }

    template <typename... Args>
    value_type coeff(Args&&... args) const {
        // rvalue type stuff will be forwarded but not moved except for in one
        // place so forwarding twice shouldn't be an issue
        return lhs_value<false>(std::make_integer_sequence<rank_type, lhs_rank>{},
                         std::forward<Args>(args)...) *
               rhs_value<true>(std::make_integer_sequence<rank_type, rhs_rank>{},
                         std::forward<Args>(args)...);

         - lhs_value<true>(std::make_integer_sequence<rank_type, lhs_rank>{},
                         std::forward<Args>(args)...) *
               rhs_value<false>(std::make_integer_sequence<rank_type, rhs_rank>{},
                         std::forward<Args>(args)...);
    }

};  // namespace binarytemplate<typenameA,typenameB>class WedgeProductView

template <concepts::ViewDerived A, concepts::ViewDerived B>
WedgeProductView(const A& a, const B& b) -> WedgeProductView<A, B>;
}  // namespace binary
}  // namespace uvl::views
#endif
