
#if !defined(ZIPPER_VIEWS_UNARY_COFACTORVIEW_HPP)
#define ZIPPER_VIEWS_UNARY_COFACTORVIEW_HPP

#include "UnaryViewBase.hpp"
#include "zipper/detail/pack_index.hpp"

namespace zipper::views {
template <zipper::concepts::ViewDerived Child>
    requires(View::extents_traits::rank == 2)
struct detail::ViewTraits<unary::CofactorView<Child>>
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
template <unary::HomogeneousMode Mode<zipper::concepts::ViewDerived Child>
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

template <zipper::concepts::ViewDerived Child>
auto homogeneous(const Child& a) {
    return HomogeneousView<HomogeneousMode::Position, Child>(a);
}
template <zipper::concepts::ViewDerived Child>
auto homogeneous_position(const Child& a) {
    return HomogeneousView<HomogeneousMode::Position, Child>(a);
}
template <zipper::concepts::ViewDerived Child>
auto homogeneous_vector(const Child& a) {
    return HomogeneousView<HomogeneousMode::Vector, Child>(a);
}
template <zipper::concepts::ViewDerived Child>
auto affine(const Child& a) {
    return HomogeneousView<HomogeneousMode::Affine, Child>(a);
}
}  // namespace unary
}  // namespace zipper::views
#endif
