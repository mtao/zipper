#if !defined(UVL_VIEWS_UNARY_SLICEVIEW_HPP)
#define UVL_VIEWS_UNARY_SLICEVIEW_HPP

#include "UnaryViewBase.hpp"
#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/detail/is_integral_constant.hpp"
#include "uvl/storage/PlainObjectStorage.hpp"
#include "uvl/views/DimensionedViewBase.hpp"

namespace uvl::views {
namespace unary {
template <concepts::ViewDerived ViewType, bool IsConst, typename... Slices>
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
        std::index_sequence<Indices...>) {
        constexpr size_t Rank = sizeof...(Indices);
        std::array<rank_type, Rank> ret;
        using tuple_type = std::tuple<Slices...>;
        size_t index = 0;
        auto add = []<std::size_t J>(std::integral_constant<std::size_t, J>,
                                     auto& ret, size_t& index) {
            if (uvl::detail::is_integral_constant_v<
                    std::tuple_element_t<J, tuple_type>>) {
                ret[J] = index++;
            }
        };
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

    constexpr static std::array<rank_type, view_extents_type::rank()>
        actionable_indices = traits::actionable_indices;

    ViewType& view()
        requires(!IsConst)
    {
        return const_cast<ViewType&>(Base::view());
    }

    SliceView(const SliceView&) = default;
    SliceView(SliceView&&) = default;
    SliceView& operator=(const SliceView&) = default;
    SliceView& operator=(SliceView&&) = default;
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

    constexpr const extents_type& extents() const { return m_extents; }

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

   private:
    template <concepts::ViewDerived V>
    void assign_direct(const V& view) {
        assert(extents() == view.extents());
        for (const auto& i : uvl::detail::all_extents_indices(extents())) {
            (*this)(i) = view(i);
        }
    }

   public:
    template <concepts::ViewDerived V>
    void assign(const V& view)
        requires(uvl::views::detail::assignable_extents<
                 typename views::detail::ViewTraits<V>::extents_type,
                 extents_type>::value)
    {
        using VTraits = views::detail::ViewTraits<V>;
        using layout_policy = uvl::default_layout_policy;
        using accessor_policy = uvl::default_accessor_policy<value_type>;
        if constexpr (VTraits::is_coefficient_consistent) {
            // TODO: check sizing
            assign_direct(view);
        } else {
            uvl::storage::PlainObjectStorage<value_type, extents_type,
                                             layout_policy, accessor_policy>
                pos(uvl::detail::convert_extents<extents_type>(view.extents()));
            pos.assign(view);
            // TODO: check sizing
            assign_direct(pos);
        }
    }

   private:
    extents_type m_extents;
    std::tuple<Slices...> m_slices;
};

}  // namespace unary
}  // namespace uvl::views
#endif
