

#if !defined(UVL_VIEWS_UNARY_SWIZZLEVIEW_HPP)
#define UVL_VIEWS_UNARY_SWIZZLEVIEW_HPP

#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/detail/swizzle_extents.hpp"
#include "uvl/views/DimensionedViewBase.hpp"

namespace uvl::views {
namespace unary {
template <concepts::ViewDerived ViewType, index_type... Indices>
class SwizzleView;

}
template <concepts::ViewDerived ViewType, index_type... Indices>
struct detail::ViewTraits<unary::SwizzleView<ViewType, Indices...>>
//: public unary::detail::CoeffWiseTraits<A, B> {
//: public detail::ViewTraits<A> {
{
    using swizzler_type = uvl::detail::ExtentsSwizzler<Indices...>;
    using Base = detail::ViewTraits<ViewType>;
    using extents_type = swizzler_type::template extents_type_swizzler_t<
        typename Base::extents_type>;
    using value_type = Base::value_type;
    using mapping_type = typename Base::mapping_type;
    constexpr static bool is_writable = Base::is_writable;
};

namespace unary {
template <concepts::ViewDerived ViewType, index_type... Indices>
class SwizzleView
    : public DimensionedViewBase<SwizzleView<ViewType, Indices...>> {
   public:
    using self_type = SwizzleView<ViewType, Indices...>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using mapping_type = traits::mapping_type;
    using value_type = traits::value_type;
    using swizzler_type = traits::swizzler_type;

    SwizzleView(const SwizzleView&) = default;
    SwizzleView(SwizzleView&&) = default;
    SwizzleView& operator=(const SwizzleView&) = default;
    SwizzleView& operator=(SwizzleView&&) = default;
    SwizzleView(const ViewType& b)
        : m_view(b), m_extents(swizzler_type::swizzle_extents(b.extents())) {}
    using Base = DimensionedViewBase<self_type>;
    using Base::extent;

    constexpr const extents_type& extents() const { return m_extents; }

    template <concepts::TupleLike T, rank_type... ranks>
    auto _coeff(const T& idxs, std::integer_sequence<rank_type, ranks...>) const
        -> value_type {
        return m_view.coeff(std::get<ranks>(idxs)...);
    }
    template <concepts::TupleLike T, rank_type... ranks>
    auto _coeff_ref(const T& idxs, std::integer_sequence<rank_type, ranks...>)
        -> value_type& requires(traits::is_writable) {
            return m_view.coeff_ref(std::get<ranks>(idxs)...);
        }

    template <concepts::TupleLike T, rank_type... ranks>
    auto _const_coeff_ref(const T& idxs,
                          std::integer_sequence<rank_type, ranks...>) const
        -> const value_type& requires(traits::is_writable) {
            return m_view.const_coeff_ref(std::get<ranks>(idxs)...);
        }

    template <typename... Args>
    value_type coeff(Args&&... idxs) const {
        return _coeff(
            swizzler_type::swizzle(std::forward<Args>(idxs)...),
            std::make_integer_sequence<rank_type, extents_type::rank()>{});
    }
    template <typename... Args>
    value_type& coeff_ref(Args&&... idxs) const
        requires(traits::is_writable)
    {
        return _coeff_ref(
            swizzler_type::swizzle(std::forward<Args>(idxs)...),
            std::make_integer_sequence<rank_type, extents_type::rank()>{});
    }
    template <typename... Args>
    const value_type& const_coeff_ref(Args&&... idxs) const
        requires(traits::is_writable)
    {
        return _const_coeff_ref(
            swizzler_type::swizzle(std::forward<Args>(idxs)...),
            std::make_integer_sequence<rank_type, extents_type::rank()>{});
    }

   private:
    const ViewType& m_view;
    extents_type m_extents;
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

}  // namespace unary
}  // namespace uvl::views
#endif
