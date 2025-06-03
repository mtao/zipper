#if !defined(ZIPPER_VIEWS_UNARY_HOMOGENEOUSVIEW_HPP)
#define ZIPPER_VIEWS_UNARY_HOMOGENEOUSVIEW_HPP

#include "UnaryViewBase.hpp"
#include "zipper/detail/pack_index.hpp"

namespace zipper::views {
namespace unary {
enum class HomogeneousMode { Position, Vector, Affine };
template <HomogeneousMode Mode, concepts::ViewDerived Child>
class HomogeneousView;

}  // namespace unary
namespace detail {
template <rank_type N, index_type Size, typename>
struct extend_extents_single_dimension;
template <rank_type N, index_type Size, index_type... Idxs>
struct extend_extents_single_dimension<N, Size, extents<Idxs...>> {
    template <typename>
    struct _ {};
    template <rank_type... M>
    struct _<std::integer_sequence<rank_type, M...>> {
        template <rank_type S>
        constexpr static index_type V() {
            return zipper::detail::pack_index<S>(Idxs...);
        }
        using type =
            extents<N == M ? (V<M>() == std::dynamic_extent ? V<M>()
                                                            : V<M>() + Size)
                           : V<M>()...>;
    };
    using __ = _<std::make_integer_sequence<rank_type, sizeof...(Idxs)>>;
    using extended_extents_type = __::type;

    using dynamic_indices_helper =
        zipper::detail::extents::DynamicExtentIndices<extended_extents_type>;
    constexpr static rank_type rank_dynamic =
        dynamic_indices_helper::rank_dynamic;

    template <std::size_t... M>
    static auto run(const zipper::extents<Idxs...>& e,
                    std::integer_sequence<std::size_t, M...>) {
        constexpr static auto dynamic_indices = dynamic_indices_helper::value;
        return std::array<index_type, rank_dynamic>{{(
            e.extent(std::get<M>(dynamic_indices)) + (M == N ? Size : 0))...}};
    }
    static auto run(const zipper::extents<Idxs...>& e) {
        return run(e, std::make_index_sequence<std::size_t(rank_dynamic)>{});
    }
};
}  // namespace detail
template <unary::HomogeneousMode Mode, concepts::ViewDerived Child>
struct detail::ViewTraits<unary::HomogeneousView<Mode, Child>>
    : public zipper::views::unary::detail::DefaultUnaryViewTraits<Child> {
    using ChildTraits = ViewTraits<Child>;
    using value_type = typename ChildTraits::value_type;
    using Base = detail::ViewTraits<Child>;

    using helper = detail::extend_extents_single_dimension<
        0, 1, typename ChildTraits::extents_type>;
    using extents_type = helper ::extended_extents_type;
    template <typename Derived>
    using base_type = DimensionedViewBase<Derived>;

    static extents_type make_extents(const ChildTraits::extents_type& a) {
        return helper::run(a);
    }
    constexpr static bool holds_extents = true;
    constexpr static bool is_writable = false;
    constexpr static bool is_coefficient_consistent = true;
    constexpr static bool is_value_based = false;
};

namespace unary {
template <unary::HomogeneousMode Mode, concepts::ViewDerived Child>
class HomogeneousView
    : public UnaryViewBase<HomogeneousView<Mode, Child>, Child> {
   public:
    using self_type = HomogeneousView<Mode, Child>;
    using traits = zipper::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    constexpr static bool is_static = extents_traits::is_static;
    using value_type = traits::value_type;

    using Base = UnaryViewBase<self_type, Child>;
    using Base::Base;
    using Base::view;

    HomogeneousView(const Child& a)
        requires(is_static)
        : Base(a) {}
    HomogeneousView(const Child& a)
        requires(!is_static)
        : Base(a, traits::make_extents(a.extents())) {}

    // using child_value_type = traits::base_value_type;

    template <typename... Args>
    value_type coeff(Args&&... args) const
    // requires(is_value_based)
    {
        if (index_type(zipper::detail::pack_index<0>(args...)) ==
            view().extent(0)) {
            if constexpr (Mode == HomogeneousMode::Position) {
                return 1;
            } else if constexpr (Mode == HomogeneousMode::Vector) {
                return 0;
            } else {
                static_assert(extents_type::rank() == 2);
                if (index_type(zipper::detail::pack_index<1>(args...)) ==
                    view().extent(1) - 1) {
                    return 1;
                } else {
                    return 0;
                }
            }
        } else {
            return view().coeff(args...);
        }
    }
};

template <concepts::ViewDerived Child>
auto homogeneous(const Child& a) {
    return HomogeneousView<HomogeneousMode::Position, Child>(a);
}
template <concepts::ViewDerived Child>
auto homogeneous_position(const Child& a) {
    return HomogeneousView<HomogeneousMode::Position, Child>(a);
}
template <concepts::ViewDerived Child>
auto homogeneous_vector(const Child& a) {
    return HomogeneousView<HomogeneousMode::Vector, Child>(a);
}
template <concepts::ViewDerived Child>
auto affine(const Child& a) {
    return HomogeneousView<HomogeneousMode::Affine, Child>(a);
}
}  // namespace unary
}  // namespace zipper::views
#endif
