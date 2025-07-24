#if !defined(ZIPPER_VIEWS_UNARY_SLICEVIEW_HPP)
#define ZIPPER_VIEWS_UNARY_SLICEVIEW_HPP

#include "UnaryViewBase.hpp"
#include "zipper/concepts/SlicePackLike.hpp"
#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/detail/is_integral_constant.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/storage/PlainObjectStorage.hpp"
#include "zipper/views/detail/AssignHelper.hpp"

namespace zipper::views {
namespace unary {
template <concepts::QualifiedViewDerived ViewType, typename... Slices>
    requires(concepts::SlicePackLike<Slices...>)
class SliceView;

}
template <concepts::QualifiedViewDerived QualifiedViewType, typename... Slices>
struct detail::ViewTraits<unary::SliceView<QualifiedViewType, Slices...>>
    : public zipper::views::unary::detail::DefaultUnaryViewTraits<
          std::decay_t<QualifiedViewType>, true> {
    using ViewType = std::decay_t<QualifiedViewType>;
    using Base = detail::ViewTraits<ViewType>;

    using extents_type =
        std::decay_t<decltype(std::experimental::submdspan_extents(
            std::declval<typename Base::extents_type>(),
            std::declval<Slices>()...))>;
    using value_type = Base::value_type;
    constexpr static bool is_const = std::is_const_v<QualifiedViewType>;
    constexpr static bool is_writable = Base::is_writable && !is_const;
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;

    //
    template <std::size_t... Indices>
    constexpr static std::array<rank_type, sizeof...(Indices)> get_actionable(
        std::index_sequence<Indices...>)
        requires(sizeof...(Indices) == Base::extents_type::rank())
    {
        constexpr size_t Rank = sizeof...(Indices);
        using tuple_type = std::tuple<Slices...>;
        constexpr auto eval =
            []<rank_type J>(std::integral_constant<rank_type, J>) -> rank_type {
            if (concepts::SliceLike<std::tuple_element_t<J, tuple_type>> &&
                !concepts::IndexLike<std::tuple_element_t<J, tuple_type>>) {
                return 0;
            } else {
                return std::dynamic_extent;
            }
        };
        std::array<rank_type, Rank> r{
            {eval(std::integral_constant<size_t, Indices>{})...}};
        rank_type index = 0;
        for (auto& v : r) {
            if (v == 0) {
                v = index++;
            }
        }
        return r;
    }

    constexpr static std::array<rank_type, Base::extents_type::rank()>
        actionable_indices = get_actionable(
            std::make_index_sequence<Base::extents_type::rank()>{});
};

namespace unary {
template <concepts::QualifiedViewDerived QualifiedViewType, typename... Slices>
    requires(concepts::SlicePackLike<Slices...>)
class SliceView : public UnaryViewBase<SliceView<QualifiedViewType, Slices...>,
                                       QualifiedViewType> {
   public:
    using self_type = SliceView<QualifiedViewType, Slices...>;
    using traits = zipper::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;
    using Base = UnaryViewBase<self_type, QualifiedViewType>;
    using Base::extent;
    using Base::view;
    constexpr static bool IsConst = traits::is_const;
    using ViewType = std::decay_t<QualifiedViewType>;
    // TODO: fix this so ViewTraits are all aware of qulaifiacitons
    using qualified_view_traits =
        zipper::views::detail::ViewTraits<QualifiedViewType>;
    using view_traits = zipper::views::detail::ViewTraits<ViewType>;
    using view_extents_type = view_traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;

    using slice_storage_type = std::tuple<Slices...>;

    constexpr static std::array<rank_type, view_extents_type::rank()>
        actionable_indices = traits::actionable_indices;

    SliceView(const SliceView&) = default;
    SliceView(SliceView&&) = default;

    SliceView& operator=(concepts::ViewDerived auto const& v) {
        assign(v);
        return *this;
    }
    /*
    SliceView(const ViewType& b, Slices&&... slices)
        requires(IsConst)
        : Base(b),
          m_extents(
              std::experimental::submdspan_extents(b.extents(), slices...)),
          m_slices(std::forward<Slices>(slices)...) {}

    SliceView(ViewType& b, Slices&&... slices)
        requires(!IsConst && view_traits::is_writable)
        : Base(b),
          m_extents(
              std::experimental::submdspan_extents(b.extents(), slices...)),
          m_slices(std::forward<Slices>(slices)...) {}

          */
    // for some reason having zipper::full_extent makes this necessary. TODO fix
    // this
    SliceView(const ViewType& b, const Slices&... slices)
        requires(IsConst)
        : Base(b, std::experimental::submdspan_extents(b.extents(), slices...)),
          m_slices(slices...) {}

    SliceView(ViewType& b, const Slices&... slices)
        requires(!IsConst && view_traits::is_writable)
        : Base(b, std::experimental::submdspan_extents(b.extents(), slices...)),
          m_slices(slices...) {}

    template <rank_type K, typename... Args>
    index_type get_index(Args&&... a) const {
        // const auto& s = std::get<K>(m_slices);
        const auto& s = std::get<K>(m_slices);

        if constexpr (zipper::detail::is_integral_constant_v<
                          std::decay_t<decltype(s)>>) {
            return s;
        } else if constexpr (std::is_integral_v<std::decay_t<decltype(s)>>) {
            return s;
        } else if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                                            zipper::full_extent_type>) {
            const auto& v =
                zipper::detail::pack_index<actionable_indices[K]>(a...);
            return v;
        } else {
            const index_type start = std::experimental::detail::first_of(s);
            const index_type stride = std::experimental::detail::stride_of(s);

            const auto& v =
                zipper::detail::pack_index<actionable_indices[K]>(a...);

            return start + v * stride;
        }
    }
    /*
    template <rank_type K>
    index_type get_index(concepts::TupleLike auto const& a) const {
        // const auto& s = std::get<K>(m_slices);
        const auto& s = std::get<K>(m_slices);
        if constexpr (zipper::detail::is_integral_constant_v<
                          std::decay_t<decltype(s)>>) {
            return s;
        } else if constexpr (std::is_integral_v<std::decay_t<decltype(s)>>) {
            return s;
        } else {
            constexpr index_type start = std::experimental::detail::first_of(s);
            constexpr index_type stride =
                std::experimental::detail::stride_of(s);

            const auto& v = std::get<actionable_indices[K]>(a);
            return start + v * stride;
        }
    }
    */

    template <typename... Args, rank_type... ranks>
    auto _coeff(std::integer_sequence<rank_type, ranks...>,
                Args&&... idxs) const -> value_type {
        // static_assert(((ranks >= sizeof...(Args)) && ...));
        return view().coeff(get_index<ranks>(idxs...)...);
    }
    template <typename... Args, rank_type... ranks>
    auto _coeff_ref(std::integer_sequence<rank_type, ranks...>, Args&&... idxs)
        -> value_type&
        requires(traits::is_writable)
    {
        // static_assert(((ranks >= sizeof...(Args)) && ...));
        return view().coeff_ref(get_index<ranks>(idxs...)...);
    }
    template <typename... Args, rank_type... ranks>
    auto _const_coeff_ref(std::integer_sequence<rank_type, ranks...>,
                          Args&&... idxs) const -> const value_type&
        requires(traits::is_writable)
    {
        // static_assert(((ranks >= sizeof...(Args)) && ...));
        return view().const_coeff_ref(get_index<ranks>(idxs...)...);
    }

    template <typename... Args>
    value_type coeff(Args&&... idxs) const {
        return _coeff(
            std::make_integer_sequence<rank_type, view_extents_type::rank()>{},
            std::forward<Args>(idxs)...);
    }
    template <typename... Args>
    value_type& coeff_ref(Args&&... idxs)
        requires(traits::is_writable)
    {
        return _coeff_ref(
            std::make_integer_sequence<rank_type, view_extents_type::rank()>{},
            std::forward<Args>(idxs)...);
    }
    template <typename... Args>
    const value_type& const_coeff_ref(Args&&... idxs) const
        requires(traits::is_writable)
    {
        return _const_coeff_ref(
            std::make_integer_sequence<rank_type, view_extents_type::rank()>{},
            std::forward<Args>(idxs)...);
    }

    template <concepts::ViewDerived V>
    void assign(const V& v)
        requires(
            traits::is_writable &&
            extents_traits::template is_convertable_from<
                typename zipper::views::detail::ViewTraits<V>::extents_type>())
    {
        views::detail::AssignHelper<V, self_type>::assign(v, *this);
    }

   private:
    slice_storage_type m_slices;
};

template <concepts::ViewDerived ViewType, typename... Slices>
SliceView(const ViewType& view, Slices&&...)
    -> SliceView<const ViewType, std::decay_t<Slices>...>;

template <concepts::ViewDerived ViewType, typename... Slices>
SliceView(ViewType& view, Slices&&...)
    -> SliceView<ViewType, std::decay_t<Slices>...>;

}  // namespace unary
}  // namespace zipper::views
#endif
