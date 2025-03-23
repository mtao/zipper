#if !defined(UVL_VIEWS_UNARY_DIAGONALVIEW_HPP)
#define UVL_VIEWS_UNARY_DIAGONALVIEW_HPP

#include "UnaryViewBase.hpp"
#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/storage/PlainObjectStorage.hpp"
#include "uvl/views/DimensionedViewBase.hpp"

namespace uvl::views {
namespace unary {
template <concepts::ViewDerived ViewType, bool IsConst>
class DiagonalView;

}
template <concepts::ViewDerived ViewType, bool IsConst>
struct detail::ViewTraits<unary::DiagonalView<ViewType, IsConst> >
    : public uvl::views::unary::detail::DefaultUnaryViewTraits<
          ViewType, DimensionedViewBase> {
    using Base = detail::ViewTraits<ViewType>;
    using value_type = Base::value_type;
    using base_extents_type = Base::extents_type;
    using base_extents_traits = uvl::detail::ExtentsTraits<base_extents_type>;
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
    using extents_type = uvl::extents<get_min_extent_static()>;
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
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;
    using Base = UnaryViewBase<self_type, ViewType>;
    using Base::extent;
    using Base::view;
    using view_traits = uvl::views::detail::ViewTraits<ViewType>;
    using view_extents_type = view_traits::extents_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;

    constexpr static std::array<rank_type, view_extents_type::rank()>
        actionable_indices = traits::actionable_indices;

    ViewType& view()
        requires(!IsConst)
    {
        return const_cast<ViewType&>(Base::view());
    }

    DiagonalView(const DiagonalView&) = default;
    DiagonalView(DiagonalView&&) = default;
    DiagonalView(const ViewType& b)
        : Base(b), m_extents(traits::get_extents(b.extents())) {}

    DiagonalView(ViewType& b)
        requires(!IsConst && view_traits::is_writable)
        : Base(b), m_extents(traits::get_extents(b.extents())) {}

    constexpr const extents_type& extents() const { return m_extents; }

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

   private:
    template <concepts::ViewDerived V>
    void assign_direct(const V& view) {
        assert(extents() == view.extents());
        for (const auto& i :
             uvl::detail::extents::all_extents_indices(extents())) {
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
};
template <concepts::ViewDerived ViewType>
DiagonalView(ViewType& v) -> DiagonalView<ViewType, false>;

template <concepts::ViewDerived ViewType>
DiagonalView(const ViewType& v) -> DiagonalView<ViewType, true>;

}  // namespace unary
}  // namespace uvl::views
#endif
