#if !defined(UVL_VIEWS_UNARY_SLICEVIEW_HPP)
#define UVL_VIEWS_UNARY_SLICEVIEW_HPP

#include "UnaryViewBase.hpp"
#include "uvl/concepts/ViewDerived.hpp"
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
          m_extents(std::experimental::submdspan_extents(
              b.extents(), std::forward<Slices>(slices)...)) {}

    SliceView(ViewType& b, Slices&&... slices)
        requires(!IsConst && view_traits::is_writable)
        : Base(b),
          m_extents(std::experimental::submdspan_extents(
              b.extents(), std::forward<Slices>(slices)...)) {}

    constexpr const extents_type& extents() const { return m_extents; }

    template <rank_type K>
    auto get_index(concepts::TupleLike auto const& a) const {
        const auto& s = std::get<K>(m_slices);
        const auto& v = std::get<K>(a);

        constexpr index_type start = std::experimental::detail::first_of(s);
        constexpr index_type stride = std::experimental::detail::stride_of(s);

        return start + v * stride;
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
            std::make_integer_sequence<rank_type, extents_type::rank()>{});
    }
    template <typename... Args>
    value_type& coeff_ref(Args&&... idxs)
        requires(traits::is_writable)
    {
        return _coeff_ref(
            std::make_tuple(std::forward<Args>(idxs)...),
            std::make_integer_sequence<rank_type, extents_type::rank()>{});
    }
    template <typename... Args>
    const value_type& const_coeff_ref(Args&&... idxs) const
        requires(traits::is_writable)
    {
        return _const_coeff_ref(
            std::make_tuple(std::forward<Args>(idxs)...),
            std::make_integer_sequence<rank_type, extents_type::rank()>{});
    }

   private:
    extents_type m_extents;
    std::tuple<Slices...> m_slices;
};

}  // namespace unary
}  // namespace uvl::views
#endif
