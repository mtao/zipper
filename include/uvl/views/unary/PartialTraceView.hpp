#include "uvl/views/reductions/CoefficientSum.hpp"
#if !defined(UVL_VIEWS_UNARY_PARTIALTRACEVIEW_HPP)
#define UVL_VIEWS_UNARY_PARTIALTRACEVIEW_HPP

#include "UnaryViewBase.hpp"
#include "detail/invert_integer_sequence.hpp"
#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/detail/pack_index.hpp"
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
    using extents_type =
        typename index_remover::template assign_types<uvl::extents>;
    using summed_extents_type = uvl::extents<Indices...>;
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
    using Base = UnaryViewBase<self_type, ViewType>;
    using Base::extent;
    using Base::view;

    PartialTraceView(const PartialTraceView&) = default;
    PartialTraceView(PartialTraceView&&) = default;
    PartialTraceView(const ViewType& b) {}
    //: Base(b), m_extents(swizzler_type::swizzle_extents(b.extents())) {}

    constexpr const extents_type& extents() const { return m_extents; }

    template <typename... Args, rank_type N>
    auto get_index(std::integral_constant<rank_type, N>, Args&&... idxs) const {
        if constexpr (((N == Indices) || ...)) {
            return uvl::full_extent;
        } else {
            return uvl::detail::pack_index<
                traits::index_remover::reversal_array[N]>(
                std::forward<Args>(idxs)...);
        }
    }

    template <typename... Args, rank_type... N>
    value_type _coeff(std::integer_sequence<rank_type, N...>,
                      Args&&... idxs) const {
        const auto slice = SliceView<ViewType, true>(
            view(), get_index(std::integral_constant<rank_type, N>{},
                              std::forward<Args>(idxs)...)...);

        auto diag = DiagonalView(slice);
        return reductions::CoefficientSum(diag)();
    }

    template <typename... Args>
    value_type coeff(Args&&... idxs) const {
        return _coeff(
            std::make_integer_sequence<rank_type, sizeof...(Indices)>{},
            std::forward<Args>(idxs)...);
    }

   private:
    extents_type m_extents;
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

}  // namespace unary
}  // namespace uvl::views
#endif
