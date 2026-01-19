#if !defined(ZIPPER_VIEWS_UNARY_DIAGONALTENSORVIEW_HPP)
#define ZIPPER_VIEWS_UNARY_DIAGONALTENSORVIEW_HPP

#include "UnaryViewBase.hpp"
#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/views/nullary/IdentityView.hpp"
#include "zipper/views/detail/AssignHelper.hpp"

namespace zipper::views {
namespace unary {
    // extracts the diagonal elements of a tensor
template <zipper::concepts::QualifiedViewDerived ViewType, rank_type N> requires (ViewType::extents_type::rank() == 1)
class DiagonalTensorView;

}
template <zipper::concepts::QualifiedViewDerived ViewType, rank_type N> requires (ViewType::extents_type::rank() == 1)
struct detail::ViewTraits<unary::DiagonalTensorView<ViewType, N> >
    : public zipper::views::unary::detail::DefaultUnaryViewTraits<
          ViewType, true> {//TODO: make this deceay less necessary
    using Base = detail::ViewTraits<ViewType>;
    using value_type = Base::value_type;
    using base_extents_type = Base::extents_type;
    using base_extents_traits = zipper::detail::ExtentsTraits<base_extents_type>;
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
    constexpr static index_type base_size = base_extents_type::static_extent(0);


    template <typename>
        struct _detail;
    template <rank_type... Ns>
    struct _detail<std::integer_sequence<rank_type,Ns...>>{

        using extents_type = extents<(Ns*0+base_size)...>;

        constexpr static extents_type make_extents(const base_extents_type& e) {
            if constexpr(base_size == std::dynamic_extent) {
                return extents_type{(Ns*0 + e.extent(0))...};
            }
        }
    };
    using detail = _detail<decltype(std::make_integer_sequence<rank_type,N>{})>;

    using extents_type = detail::extents_type;
    constexpr extents_type make_extents(const base_extents_type& e) {
        return detail::make_extents(e);
    }
    //
};

namespace unary {
template <zipper::concepts::QualifiedViewDerived ViewType, rank_type N> requires (ViewType::extents_type::rank() == 1)
class DiagonalTensorView
    : public UnaryViewBase<DiagonalTensorView<ViewType, N>, ViewType> {
   public:
    using self_type = DiagonalTensorView<ViewType, N>;
    using traits = zipper::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;
    using Base = UnaryViewBase<self_type, ViewType>;
    using Base::extent;
    using Base::view;
    using view_traits = traits::Base;
    using view_extents_type = view_traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    constexpr static bool holds_extents = traits::holds_extents;
    static_assert(holds_extents);



    DiagonalTensorView(const DiagonalTensorView& o): DiagonalTensorView(o.view()) {}
    DiagonalTensorView(DiagonalTensorView&&o): DiagonalTensorView(o.view()) {}

    DiagonalTensorView& operator=(const DiagonalTensorView&) = delete;
    DiagonalTensorView& operator=(DiagonalTensorView&&) = delete;

    DiagonalTensorView(ViewType& b)
        : Base(b, traits::make_extents(b.extents())) {}


    template <rank_type K>
    index_type get_index(zipper::concepts::TupleLike auto const& a) const {
        return std::get<0>(a);
    }

    template <zipper::concepts::TupleLike T, rank_type... ranks>
    auto _coeff(const T& idxs, std::integer_sequence<rank_type, ranks...>) const
        -> value_type {
        return view().coeff(get_index<ranks>(idxs)...);
    }
    template <zipper::concepts::TupleLike T, rank_type... ranks>
    auto _coeff_ref(const T& idxs, std::integer_sequence<rank_type, ranks...>)
        -> value_type& requires(traits::is_writable) {
            return view().coeff_ref(get_index<ranks>(idxs)...);
        }

    template <zipper::concepts::TupleLike T, rank_type... ranks>
    auto _const_coeff_ref(const T& idxs,
                          std::integer_sequence<rank_type, ranks...>) const
        -> const value_type& requires(traits::is_writable) {
            return view().const_coeff_ref(get_index<ranks>(idxs)...);
        }

    template <typename... Args>
    value_type coeff(Args&&... idxs) const {
        return _coeff(
            std::make_tuple(std::forward<Args>(idxs)...),
            std::make_integer_sequence<rank_type, view_extents_type::rank()>{});
    }
    template <typename... Args>
    value_type& coeff_ref(Args&&... idxs)
        requires(traits::is_writable)
    {
        return _coeff_ref(
            std::make_tuple(std::forward<Args>(idxs)...),
            std::make_integer_sequence<rank_type, view_extents_type::rank()>{});
    }
    template <typename... Args>
    const value_type& const_coeff_ref(Args&&... idxs) const
        requires(traits::is_writable)
    {
        return _const_coeff_ref(
            std::make_tuple(std::forward<Args>(idxs)...),
            std::make_integer_sequence<rank_type, view_extents_type::rank()>{});
    }

    template <zipper::concepts::ViewDerived V>
    void assign(const V& v)
        requires(
            traits::is_writable &&
            extents_traits::template is_convertable_from<
                typename zipper::views::detail::ViewTraits<V>::extents_type>())
    {
        views::detail::AssignHelper<V, self_type>::assign(v, *this);
    }

};

template <zipper::concepts::QualifiedViewDerived ViewType, rank_type N>
DiagonalTensorView(const ViewType& v) -> DiagonalTensorView<ViewType, rank_type N>;

}  // namespace unary
}  // namespace zipper::views
#endif
