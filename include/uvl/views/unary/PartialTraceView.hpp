#if !defined(UVL_VIEWS_UNARY_PARTIALTRACEVIEW_HPP)
#define UVL_VIEWS_UNARY_PARTIALTRACEVIEW_HPP

#include "UnaryViewBase.hpp"
#include "detail/invert_integer_sequence.hpp"
#include "uvl/concepts/ViewDerived.hpp"
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
    using Base = detail::ViewTraits<ViewType>;
    using index_remover =
        unary::detail::invert_integer_sequence<Base::extents_type::rank(),
                                               Indices...>;
    using extents_type = typename index_remover::template assign_types<uvl::extents>;
    using value_type = Base::value_type;
    constexpr static bool is_writable = false;
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

    PartialTraceView(const PartialTraceView&) = default;
    PartialTraceView(PartialTraceView&&) = default;
    PartialTraceView& operator=(const PartialTraceView&) = default;
    PartialTraceView& operator=(PartialTraceView&&) = default;
    PartialTraceView(const ViewType& b)
        : Base(b), m_extents(swizzler_type::swizzle_extents(b.extents())) {}

    constexpr const extents_type& extents() const { return m_extents; }

    template <concepts::TupleLike T, rank_type... ranks>
    auto _coeff(const T& idxs, std::integer_sequence<rank_type, ranks...>) const
        -> value_type {
        return view().coeff(std::get<ranks>(idxs)...);
    }
    template <typename... Args>
    value_type coeff(Args&&... idxs) const {
        return _coeff(
            swizzler_type::swizzle(std::forward<Args>(idxs)...),
            std::make_integer_sequence<rank_type, extents_type::rank()>{});
    }

   private:
    extents_type m_extents;
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

}  // namespace unary
}  // namespace uvl::views
#endif
