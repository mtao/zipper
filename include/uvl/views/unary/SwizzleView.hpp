

#if !defined(UVL_VIEWS_UNARY_SWIZZLEVIEW_HPP)
#define UVL_VIEWS_UNARY_SWIZZLEVIEW_HPP

#include "UnaryViewBase.hpp"
#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/detail/extents/all_extents_indices.hpp"
#include "uvl/detail/extents/swizzle_extents.hpp"
#include "uvl/storage/PlainObjectStorage.hpp"
#include "uvl/views/DimensionedViewBase.hpp"

namespace uvl::views {
namespace unary {
template <concepts::ViewDerived ViewType, index_type... Indices>
class SwizzleView;

}
template <concepts::ViewDerived ViewType, index_type... Indices>
struct detail::ViewTraits<unary::SwizzleView<ViewType, Indices...>>
    : public uvl::views::unary::detail::DefaultUnaryViewTraits<
          ViewType, DimensionedViewBase> {
    using swizzler_type = uvl::detail::extents::ExtentsSwizzler<Indices...>;
    using Base = detail::ViewTraits<ViewType>;
    using extents_type = swizzler_type::template extents_type_swizzler_t<
        typename Base::extents_type>;
    using value_type = Base::value_type;
    constexpr static bool is_writable = Base::is_writable;
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
};

namespace unary {
template <concepts::ViewDerived ViewType, index_type... Indices>
class SwizzleView
    : public UnaryViewBase<SwizzleView<ViewType, Indices...>, ViewType> {
   public:
    using self_type = SwizzleView<ViewType, Indices...>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;
    using extents_traits = uvl::detail::ExtentsTraits<extents_type>;
    using swizzler_type = traits::swizzler_type;
    using Base = UnaryViewBase<self_type, ViewType>;
    using Base::extent;
    using Base::view;
    constexpr static rank_type internal_rank = ViewType::extents_type::rank();
    constexpr static std::array<rank_type, internal_rank>
        to_internal_rank_indices = swizzler_type::valid_internal_indices;

    SwizzleView(const SwizzleView&) = default;
    SwizzleView(SwizzleView&&) = default;
    SwizzleView& operator=(const SwizzleView&) = default;
    SwizzleView& operator=(SwizzleView&&) = default;
    SwizzleView(const ViewType& b)
        : Base(b), m_extents(swizzler_type::swizzle_extents(b.extents())) {}

    constexpr const extents_type& extents() const { return m_extents; }

    template <concepts::TupleLike T, rank_type... ranks>
    auto _coeff(const T& idxs, std::integer_sequence<rank_type, ranks...>) const
        -> value_type {
        return view().coeff(std::get<ranks>(idxs)...);
    }
    template <concepts::TupleLike T, rank_type... ranks>
    auto _coeff_ref(const T& idxs, std::integer_sequence<rank_type, ranks...>)
        -> value_type& requires(traits::is_writable) {
            return view().coeff_ref(std::get<ranks>(idxs)...);
        }

    template <concepts::TupleLike T, rank_type... ranks>
    auto _const_coeff_ref(const T& idxs,
                          std::integer_sequence<rank_type, ranks...>) const
        -> const value_type& requires(traits::is_writable) {
            return view().const_coeff_ref(std::get<ranks>(idxs)...);
        }

    template <typename... Args>
    value_type coeff(Args&&... idxs) const {
        return _coeff(swizzler_type::unswizzle(std::forward<Args>(idxs))...,
                      std::make_integer_sequence<rank_type, internal_rank>{});
    }
    template <typename... Args>
    value_type& coeff_ref(Args&&... idxs)
        requires(traits::is_writable)
    {
        return _coeff_ref(
            swizzler_type::unswizzle(std::forward<Args>(idxs)...),
            std::make_integer_sequence<rank_type, internal_rank>{});
    }
    template <typename... Args>
    const value_type& const_coeff_ref(Args&&... idxs) const
        requires(traits::is_writable)
    {
        return _const_coeff_ref(
            swizzler_type::unswizzle(std::forward<Args>(idxs)...),
            std::make_integer_sequence<rank_type, internal_rank>{});
    }

   private:
    template <concepts::ViewDerived V>
    void assign_direct(const V& view)
        requires(traits::is_writable)
    {
        for (const auto& i :
             uvl::detail::extents::all_extents_indices(extents())) {
            (*this)(i) = view(i);
        }
    }

   public:
    template <concepts::ViewDerived V>
    void assign(const V& view)
        requires(
            traits::is_writable &&
            extents_traits::template is_convertable_from<
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
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

}  // namespace unary
}  // namespace uvl::views
#endif
