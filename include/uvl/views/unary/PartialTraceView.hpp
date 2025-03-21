#if !defined(UVL_VIEWS_UNARY_PARTIALTRACEVIEW_HPP)
#define UVL_VIEWS_UNARY_PARTIALTRACEVIEW_HPP

#include "UnaryViewBase.hpp"
#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/detail/swizzle_extents.hpp"
#include "uvl/views/DimensionedViewBase.hpp"

namespace uvl::views {
namespace unary {
template <concepts::ViewDerived ViewType, index_type... Indices>
class PartialTraceView;

}
template <concepts::ViewDerived ViewType, index_type... Indices>
struct detail::ViewTraits<unary::PartialTraceView<ViewType, Indices...>>
    : public uvl::views::unary::detail::DefaultUnaryViewTraits<
          ViewType, DimensionedViewBase> {
    using swizzler_type = uvl::detail::ExtentsPartialTracer<Indices...>;
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
class PartialTraceView
    : public UnaryViewBase<PartialTraceView<ViewType, Indices...>, ViewType> {
   public:
    using self_type = PartialTraceView<ViewType, Indices...>;
    using traits = uvl::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;
    using swizzler_type = traits::swizzler_type;
    using Base = UnaryViewBase<self_type, ViewType>;
    using Base::extent;
    using Base::view;

    PartialTraceView(const SwizzleView&) = default;
    PartialTraceView(SwizzleView&&) = default;
    PartialTraceView& operator=(const SwizzleView&) = default;
    PartialTraceView& operator=(SwizzleView&&) = default;
    PartialTraceView(const ViewType& b)
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
        return _coeff(
            swizzler_type::swizzle(std::forward<Args>(idxs)...),
            std::make_integer_sequence<rank_type, extents_type::rank()>{});
    }
    template <typename... Args>
    value_type& coeff_ref(Args&&... idxs)
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
    extents_type m_extents;
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

}  // namespace unary
}  // namespace uvl::views
#endif
