#include "uvl/views/reductions/CoefficientSum.hpp"
#if !defined(UVL_VIEWS_UNARY_PARTIALTRACEVIEW_HPP)
#define UVL_VIEWS_UNARY_PARTIALTRACEVIEW_HPP


#include <spdlog/spdlog.h>
#include "UnaryViewBase.hpp"
#include "detail/invert_integer_sequence.hpp"
#include "uvl/concepts/ViewDerived.hpp"
#include "uvl/detail/extents/extents_formatter.hpp"
#include "uvl/detail/extents/static_extents_to_array.hpp"
#include "uvl/detail/pack_index.hpp"
#include "uvl/views/DimensionedViewBase.hpp"
#include "DiagonalView.hpp"

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
    using extents_type = typename index_remover::template assign_types<
        uvl::extents, uvl::detail::extents::static_extents_to_array_v<
                          typename ViewType::extents_type>>;
    using summed_extents_type = uvl::extents<Indices...>;
    using value_type = Base::value_type;
    constexpr static bool is_writable = false;
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
};

namespace unary {
// indices are the indices being traced
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
    PartialTraceView(const ViewType& b) : Base(b) {}
    //: Base(b), m_extents(swizzler_type::swizzle_extents(b.extents())) {}

    constexpr const extents_type& extents() const { return m_extents; }

    template <typename>
    struct slice_type_;
    template <rank_type... N>
    struct slice_type_<std::integer_sequence<rank_type, N...>> {
        using type =
            SliceView<ViewType, true,
                      std::conditional_t<traits::index_remover::in_sequence(N),
                                         index_type, full_extent_type>...>;
    };

    using slice_type = slice_type_<std::decay_t<
        decltype(std::make_integer_sequence<
                 rank_type, ViewType::extents_type::rank()>{})>>::type;

    template <typename... Args, rank_type N>
    auto get_index(std::integral_constant<rank_type, N>, Args&&... idxs) const {
        if constexpr (((N == Indices) || ...)) {
            return uvl::full_extent;
        } else {
            constexpr static rank_type Index =
                traits::index_remover::full_rank_to_reduced_indices[N];
            static_assert(Index <= sizeof...(Args));
            return uvl::detail::pack_index<Index>(std::forward<Args>(idxs)...);
        }
    }

    template <typename... Args, rank_type... N>
    value_type _coeff(std::integer_sequence<rank_type, N...>,
                      Args&&... idxs) const
        requires(sizeof...(N) == ViewType::extents_type::rank())
    {
        const auto slice =
            slice_type(view(), get_index(std::integral_constant<rank_type, N>{},
                                         std::forward<Args>(idxs)...)...);

        DiagonalView<slice_type, true> diag(slice);
        spdlog::info("Computing stuff! {}: {} {} {}", diag.extents(), diag(0), diag(1), diag(2));
        return reductions::CoefficientSum(diag)();
    }

    template <typename... Args>
    value_type coeff(Args&&... idxs) const {
        uvl::detail::extents::indices_in_range(extents(), idxs...);
        if constexpr (sizeof...(Indices) == 0) {
            return view().coeff(std::forward<Args>(idxs)...);
        } else {
            return _coeff(
                std::make_integer_sequence<rank_type,
                                           ViewType::extents_type::rank()>{},
                std::forward<Args>(idxs)...);
        }
    }

   private:
    extents_type m_extents;
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

}  // namespace unary
}  // namespace uvl::views
#endif
