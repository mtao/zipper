#if !defined(ZIPPER_VIEWS_BINARY_TENSORPRODUCTVIEW_HPP)
#define ZIPPER_VIEWS_BINARY_TENSORPRODUCTVIEW_HPP

#include "BinaryViewBase.hpp"
#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/detail/pack_index.hpp"

namespace zipper::views {
namespace binary {
template <zipper::concepts::QualifiedViewDerived A,
          zipper::concepts::QualifiedViewDerived B>
class TensorProductView;

namespace detail {

template <typename A, typename B>
struct tensor_coeffwise_extents_values;
template <index_type... A, index_type... B>
struct tensor_coeffwise_extents_values<extents<A...>, extents<B...>> {
    using product_extents_type = zipper::extents<A..., B...>;
    using a_extents_type = extents<A...>;
    using b_extents_type = extents<B...>;

    static_assert(product_extents_type::rank() ==
                  a_extents_type::rank() + b_extents_type::rank());

    using a_extents_traits = zipper::detail::ExtentsTraits<a_extents_type>;
    using b_extents_traits = zipper::detail::ExtentsTraits<b_extents_type>;

    template <rank_type... N, rank_type... M>
    constexpr static product_extents_type merge(
        const extents<A...>& a, const extents<B...>& b,
        std::integer_sequence<rank_type, N...>,
        std::integer_sequence<rank_type, M...>) {
        using AEI = a_extents_traits::dynamic_indices_helper;
        using BEI = b_extents_traits::dynamic_indices_helper;

        constexpr static auto ADI = AEI::dynamic_indices;
        constexpr static auto BDI = BEI::dynamic_indices;


        return product_extents_type(a.extent(ADI[N])..., b.extent(BDI[M])...);
    }

    constexpr static product_extents_type merge(const extents<A...>& a,
                                                const extents<B...>& b) {
        return merge(
            a, b,
            std::make_integer_sequence<rank_type, a_extents_type::rank_dynamic()>{},
            std::make_integer_sequence<rank_type, b_extents_type::rank_dynamic()>{}

            );
    }
};

}  // namespace detail
}  // namespace binary
template <typename A, typename B>
struct detail::ViewTraits<binary::TensorProductView<A, B>>
    : public binary::detail::DefaultBinaryViewTraits<A, B>
//: public binary::detail::WiseTraits<A, B> {
//: public detail::ViewTraits<A> {
{
    using ATraits = views::detail::ViewTraits<A>;
    constexpr static rank_type lhs_rank = ATraits::extents_type::rank();
    using BTraits = views::detail::ViewTraits<B>;
    constexpr static rank_type rhs_rank = BTraits::extents_type::rank();
    using CEV = binary::detail::tensor_coeffwise_extents_values<
        typename ATraits::extents_type, typename BTraits::extents_type>;

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
template <zipper::concepts::QualifiedViewDerived A,
          zipper::concepts::QualifiedViewDerived B>
class TensorProductView : public BinaryViewBase<TensorProductView<A, B>, A, B> {
   public:
    using self_type = TensorProductView<A, B>;
    using ViewBase<self_type>::operator();
    using traits = zipper::views::detail::ViewTraits<self_type>;
    using value_type = traits::value_type;
    using Base = BinaryViewBase<self_type, A, B>;
    using Base::extent;
    using Base::lhs;
    using Base::rhs;
    constexpr static rank_type lhs_rank = traits::lhs_rank;
    constexpr static rank_type rhs_rank = traits::rhs_rank;

    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    TensorProductView(const TensorProductView&) = default;
    TensorProductView(TensorProductView&&) = default;
    TensorProductView& operator=(const TensorProductView&) = delete;
    TensorProductView& operator=(TensorProductView&&) = delete;
    TensorProductView(const A& a, const B& b)
        requires(extents_traits::is_static)
        : Base(a, b) {}
    TensorProductView(const A& a, const B& b)
        requires(!extents_traits::is_static)
        : Base(a, b, traits::CEV::merge(a.extents(), b.extents())) {}

    template <typename... Args, rank_type... ranks>
    auto lhs_value(std::integer_sequence<rank_type, ranks...>,
                   Args&&... args) const -> decltype(auto) {
        return lhs()(
            zipper::detail::pack_index<ranks>(std::forward<Args>(args)...)...);
    }
    template <typename... Args, rank_type... ranks>
    auto rhs_value(std::integer_sequence<rank_type, ranks...>,
                   Args&&... args) const -> decltype(auto) {
        return rhs()(zipper::detail::pack_index<ranks + lhs_rank>(
            std::forward<Args>(args)...)...);
    }

    template <typename... Args>
    value_type coeff(Args&&... args) const {
        // rvalue type stuff will be forwarded but not moved except for in one
        // place so forwarding twice shouldn't be an issue
        return lhs_value(std::make_integer_sequence<rank_type, lhs_rank>{},
                         std::forward<Args>(args)...) *
               rhs_value(std::make_integer_sequence<rank_type, rhs_rank>{},
                         std::forward<Args>(args)...);
    }

};  // namespace binarytemplate<typenameA,typenameB>class TensorProductView

template <zipper::concepts::ViewDerived A, zipper::concepts::ViewDerived B>
TensorProductView(const A& a, const B& b) -> TensorProductView<A, B>;
}  // namespace binary
}  // namespace zipper::views
#endif
