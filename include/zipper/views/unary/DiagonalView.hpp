#if !defined(ZIPPER_VIEWS_UNARY_DIAGONALVIEW_HPP)
#define ZIPPER_VIEWS_UNARY_DIAGONALVIEW_HPP

#include "UnaryViewBase.hpp"
#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/storage/PlainObjectStorage.hpp"
#include "zipper/views/DimensionedViewBase.hpp"
#include "zipper/views/detail/AssignHelper.hpp"

namespace zipper::views {
namespace unary {
template <concepts::ViewDerived ViewType, bool IsConst>
class DiagonalView;

}
template <concepts::ViewDerived ViewType, bool IsConst>
struct detail::ViewTraits<unary::DiagonalView<ViewType, IsConst> >
    : public zipper::views::unary::detail::DefaultUnaryViewTraits<
          ViewType, true> {
    using Base = detail::ViewTraits<ViewType>;
    using value_type = Base::value_type;
    using base_extents_type = Base::extents_type;
    using base_extents_traits = zipper::detail::ExtentsTraits<base_extents_type>;
    constexpr static bool is_writable = Base::is_writable && !IsConst;
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
    constexpr static bool is_const = IsConst;

    //
    template <std::size_t... Indices>
    constexpr static index_type get_min_extent_static(
        std::integer_sequence<index_type, Indices...>) {
        if constexpr (sizeof...(Indices) == 1) {
            return 1;
        } else {
            return std::min({base_extents_type::static_extent(Indices)...});
        }
    }

    constexpr static index_type get_min_extent_static() {
        if constexpr (base_extents_traits::is_dynamic) {
            return std::dynamic_extent;
        } else {
            return get_min_extent_static(
                std::make_integer_sequence<index_type,
                                           base_extents_type::rank()>{});
        }
    }
    static index_type get_min_extent(const base_extents_type& e) {
        if constexpr (base_extents_traits::is_static) {
            return get_min_extent_static();
        } else if constexpr (base_extents_type::rank() == 1) {
            return std::min<index_type>(0, e.extent(0));
        } else {
            index_type min = std::numeric_limits<index_type>::max();
            ;
            for (rank_type j = 0; j < e.rank(); ++j) {
                min = std::min(min, e.extent(j));
            }
            return min;
        }
    }
    using extents_type = zipper::extents<get_min_extent_static()>;
    static extents_type get_extents(const base_extents_type& e) {
        if constexpr (base_extents_traits::is_static) {
            return {};
        } else {
            return extents_type(get_min_extent(e));
        }
    }
};

namespace unary {
template <concepts::ViewDerived ViewType, bool IsConst>
class DiagonalView
    : public UnaryViewBase<DiagonalView<ViewType, IsConst>, ViewType> {
   public:
    using self_type = DiagonalView<ViewType, IsConst>;
    using traits = zipper::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;
    using Base = UnaryViewBase<self_type, ViewType>;
    using Base::extent;
    using Base::view;
    using view_traits = zipper::views::detail::ViewTraits<ViewType>;
    using view_extents_type = view_traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    constexpr static bool holds_extents = traits::holds_extents;
    static_assert(holds_extents);

    constexpr static std::array<rank_type, view_extents_type::rank()>
        actionable_indices = traits::actionable_indices;


    DiagonalView(const DiagonalView& o): DiagonalView(o.view()) {}
    DiagonalView(DiagonalView&&o): DiagonalView(o.view()) {}

    DiagonalView& operator=(const DiagonalView&) = delete;
    DiagonalView& operator=(DiagonalView&&) = delete;
    DiagonalView(const ViewType& b)
        : Base(b, traits::get_extents(b.extents())) {}

    DiagonalView(ViewType& b)
        requires(!IsConst && view_traits::is_writable)
        : Base(b, traits::get_extents(b.extents())) {}


    template <rank_type K>
    index_type get_index(concepts::TupleLike auto const& a) const {
        return std::get<0>(a);
    }

    template <concepts::TupleLike T, rank_type... ranks>
    auto _coeff(const T& idxs, std::integer_sequence<rank_type, ranks...>) const
        -> value_type {
        return view().coeff(get_index<ranks>(idxs)...);
    }
    template <concepts::TupleLike T, rank_type... ranks>
    auto _coeff_ref(const T& idxs, std::integer_sequence<rank_type, ranks...>)
        -> value_type& requires(traits::is_writable) {
            return view().coeff_ref(get_index<ranks>(idxs)...);
        }

    template <concepts::TupleLike T, rank_type... ranks>
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

    template <concepts::ViewDerived V>
    void assign(const V& view)
        requires(
            traits::is_writable &&
            extents_traits::template is_convertable_from<
                typename zipper::views::detail::ViewTraits<V>::extents_type>())
    {
        views::detail::AssignHelper<V, self_type>::assign(view, *this);
    }

};
template <concepts::ViewDerived ViewType>
DiagonalView(ViewType& v) -> DiagonalView<ViewType, false>;

template <concepts::ViewDerived ViewType>
DiagonalView(const ViewType& v) -> DiagonalView<ViewType, true>;

}  // namespace unary
}  // namespace zipper::views
#endif
