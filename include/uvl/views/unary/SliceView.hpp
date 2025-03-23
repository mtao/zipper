#if !defined(UVL_VIEWS_UNARY_SLICEVIEW_HPP)
#define UVL_VIEWS_UNARY_SLICEVIEW_HPP

#include "UnaryViewBase.hpp"
#include "uvl/concepts/SlicePackLike.hpp"
#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/detail/is_integral_constant.hpp"
#include "uvl/detail/pack_index.hpp"
#include "uvl/storage/PlainObjectStorage.hpp"
#include "uvl/views/DimensionedViewBase.hpp"

namespace uvl::views {
namespace unary {
template <concepts::ViewDerived ViewType, bool IsConst, typename... Slices>
    requires(concepts::SlicePackLike<Slices...>)
class SliceView;

}
template <concepts::ViewDerived ViewType, bool IsConst, typename... Slices>
struct detail::ViewTraits<unary::SliceView<ViewType, IsConst, Slices...>>
    : public uvl::views::unary::detail::DefaultUnaryViewTraits<
          ViewType, DimensionedViewBase> {
    using Base = detail::ViewTraits<ViewType>;
    using extents_type =
        std::decay_t<decltype(std::experimental::submdspan_extents(
            std::declval<typename Base::extents_type>(),
            std::declval<Slices>()...))>;
    using value_type = Base::value_type;
    constexpr static bool is_writable = Base::is_writable && !IsConst;
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
    constexpr static bool is_const = IsConst;

    //
    template <std::size_t... Indices>
    constexpr static std::array<rank_type, sizeof...(Indices)> get_actionable(
        std::index_sequence<Indices...>)
        requires(sizeof...(Indices) == Base::extents_type::rank())
    {
        constexpr size_t Rank = sizeof...(Indices);
        std::array<rank_type, Rank> ret;
        using tuple_type = std::tuple<Slices...>;
        size_t index = 0;
        auto add = []<std::size_t J>(std::integral_constant<std::size_t, J>,
                                     auto& ret, size_t& index) {
            if (concepts::SliceLike<std::tuple_element_t<J, tuple_type>> &&
                !concepts::IndexLike<std::tuple_element_t<J, tuple_type>>) {
                ret[J] = index++;
            } else {
                ret[J] = std::dynamic_extent;
            }
        };
        // auto add = []<std::size_t J>(std::integral_constant<std::size_t, J>,
        //                              auto& ret, size_t& index) {
        //     if (uvl::detail::is_integral_constant_v<
        //             std::tuple_element_t<J, tuple_type>>) {
        //         ret[J] = index++;
        //     }
        // };
        ((add(std::integral_constant<std::size_t, Indices>{}, ret, index),
          ...));

        return ret;
    }

    constexpr static std::array<rank_type, Base::extents_type::rank()>
        actionable_indices = get_actionable(
            std::make_index_sequence<Base::extents_type::rank()>{});
};

namespace unary {
template <concepts::ViewDerived ViewType, bool IsConst, typename... Slices>
    requires(concepts::SlicePackLike<Slices...>)
class SliceView
    : public UnaryViewBase<SliceView<ViewType, IsConst, Slices...>, ViewType> {
   public:
    using self_type = SliceView<ViewType, IsConst, Slices...>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;
    using Base = UnaryViewBase<self_type, ViewType>;
    using Base::extent;
    using Base::view;
    using view_traits = uvl::views::detail::ViewTraits<ViewType>;
    using view_extents_type = view_traits::extents_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;

    using slice_storage_type = std::tuple<Slices...>;

    constexpr static std::array<rank_type, view_extents_type::rank()>
        actionable_indices = traits::actionable_indices;

    ViewType& view()
        requires(!IsConst)
    {
        return const_cast<ViewType&>(Base::view());
    }

    SliceView(const SliceView&) = default;
    SliceView(SliceView&&) = default;

    SliceView& operator=(concepts::ViewDerived auto const& v) {
        assign(v);
        return *this;
    }
    SliceView(const ViewType& b, Slices&&... slices)
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

    // for some reason having uvl::full_extent makes this necessary. TODO fix
    // this
    SliceView(const ViewType& b, const Slices&... slices)
        : Base(b),
          m_extents(
              std::experimental::submdspan_extents(b.extents(), slices...)),
          m_slices(slices...) {}

    SliceView(ViewType& b, const Slices&... slices)
        requires(!IsConst && view_traits::is_writable)
        : Base(b),
          m_extents(
              std::experimental::submdspan_extents(b.extents(), slices...)),
          m_slices(slices...) {}

    constexpr const extents_type& extents() const { return m_extents; }

    template <rank_type K, typename... Args>
    index_type get_index(Args&&... a) const {
        // const auto& s = std::get<K>(m_slices);
        const auto& s = std::get<K>(m_slices);

        if constexpr (uvl::detail::is_integral_constant_v<
                          std::decay_t<decltype(s)>>) {
            return s;
        } else if constexpr (std::is_integral_v<std::decay_t<decltype(s)>>) {
            return s;
        } else if constexpr (std::is_same_v<std::decay_t<decltype(s)>,
                                            uvl::full_extent_type>) {
            const auto& v =
                uvl::detail::pack_index<actionable_indices[K]>(a...);
            return v;
        } else {
            constexpr index_type start = std::experimental::detail::first_of(s);
            constexpr index_type stride =
                std::experimental::detail::stride_of(s);

            const auto& v =
                uvl::detail::pack_index<actionable_indices[K]>(a...);
            return start + v * stride;
        }
    }
    /*
    template <rank_type K>
    index_type get_index(concepts::TupleLike auto const& a) const {
        // const auto& s = std::get<K>(m_slices);
        const auto& s = std::get<K>(m_slices);
        if constexpr (uvl::detail::is_integral_constant_v<
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
        return view().coeff(get_index<ranks>(idxs...)...);
    }
    template <typename... Args, rank_type... ranks>
    auto _coeff_ref(std::integer_sequence<rank_type, ranks...>, Args&&... idxs)
        -> value_type& {
        return view().coeff_ref(get_index<ranks>(idxs...)...);
    }
    template <typename... Args, rank_type... ranks>
    auto _const_coeff_ref(std::integer_sequence<rank_type, ranks...>,
                          Args&&... idxs) const -> const value_type& {
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

   private:
    template <concepts::ViewDerived V>
    void assign_direct(const V& view) {
        for (const auto& i : uvl::detail::extents::all_extents_indices(extents())) {
            (*this)(i) = view(i);
        }
    }

   public:
    template <concepts::ViewDerived V>
    void assign(const V& view)
        requires(extents_traits::template is_convertable_from<
                 typename uvl::views::detail::ViewTraits<V>::extents_type>())
    {
        using VTraits = views::detail::ViewTraits<V>;
        using layout_policy = uvl::default_layout_policy;
        using accessor_policy = uvl::default_accessor_policy<value_type>;
#if !defined(NDEBUG)
        constexpr static bool assigning_from_infinite =
            VTraits::extents_type::rank() == 0;

        assert(assigning_from_infinite || extents() == view.extents());
#endif
        if constexpr (VTraits::is_coefficient_consistent) {
            // TODO: check sizing
            assign_direct(view);
        } else {
            uvl::storage::PlainObjectStorage<value_type, extents_type,
                                             layout_policy, accessor_policy>
                pos(extents_traits::convert_from(view.extents()));
            pos.assign(view);
            // TODO: check sizing
            assign_direct(pos);
        }
    }

   private:
    extents_type m_extents;
    slice_storage_type m_slices;
};

template <concepts::ViewDerived ViewType, typename... Slices>
SliceView(const ViewType& view, Slices&&...)
    -> SliceView<ViewType, true, std::decay_t<Slices>...>;

template <concepts::ViewDerived ViewType, typename... Slices>
SliceView(ViewType& view, Slices&&...)
    -> SliceView<ViewType, false, std::decay_t<Slices>...>;

}  // namespace unary
}  // namespace uvl::views
#endif
