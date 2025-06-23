#if !defined(ZIPPER_VIEWS_UNARY_PARTIALTRACEVIEW_HPP)
#define ZIPPER_VIEWS_UNARY_PARTIALTRACEVIEW_HPP

#include <spdlog/spdlog.h>

#include "DiagonalView.hpp"
#include "UnaryViewBase.hpp"
#include "detail/invert_integer_sequence.hpp"
#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/detail/extents/extents_formatter.hpp"
#include "zipper/detail/extents/static_extents_to_array.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/views/reductions/CoefficientSum.hpp"

namespace zipper::views {
namespace unary {
template <concepts::ViewDerived ViewType, rank_type... Indices>
    requires(sizeof...(Indices) % 2 == 0)
class PartialTraceView;

}
template <concepts::ViewDerived ViewType, rank_type... Indices>
struct detail::ViewTraits<unary::PartialTraceView<ViewType, Indices...>>
    : public zipper::views::unary::detail::DefaultUnaryViewTraits<ViewType,
                                                                  true> {
    using Base = detail::ViewTraits<ViewType>;
    using index_remover =
        unary::detail::invert_integer_sequence<Base::extents_type::rank(),
                                               Indices...>;
    using extents_type = typename index_remover::template assign_types<
        zipper::extents, zipper::detail::extents::static_extents_to_array_v<
                             typename ViewType::extents_type>>;
    using summed_extents_type = zipper::extents<Indices...>;
    using value_type = Base::value_type;
    constexpr static bool is_writable = false;
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
};

namespace unary {
// indices are the indices being traced
template <concepts::ViewDerived ViewType, rank_type... Indices>
    requires(sizeof...(Indices) % 2 == 0)
class PartialTraceView
    : public UnaryViewBase<PartialTraceView<ViewType, Indices...>, ViewType> {
   public:
    using self_type = PartialTraceView<ViewType, Indices...>;
    using traits = zipper::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;
    using Base = UnaryViewBase<self_type, ViewType>;
    using Base::extent;
    using Base::view;

    PartialTraceView(const ViewType& b)
        : Base(b, traits::index_remover::get_extents(b.extents())) {}
    PartialTraceView() = delete;
    PartialTraceView& operator=(const PartialTraceView&) = delete;
    PartialTraceView& operator=(PartialTraceView&&) = delete;
    PartialTraceView(PartialTraceView&& o) = default;
    PartialTraceView(const PartialTraceView& o) = default;
    // PartialTraceView(const PartialTraceView& o) = default;//:
    // Base(o.view(),traits::index_remover::get_extents(o.extents())){ }
    // PartialTraceView(PartialTraceView&& o): PartialTraceView(o.view()) { }
    //: Base(b), m_extents(swizzler_type::swizzle_extents(b.extents())) {}

    template <typename>
    struct slice_type_;
    template <rank_type... N>
    struct slice_type_<std::integer_sequence<rank_type, N...>> {
        using type =
            SliceView<ViewType, true,
                      std::conditional_t<traits::index_remover::in_sequence(N),
                                         rank_type, full_extent_type>...>;
    };

    using slice_type = slice_type_<std::decay_t<
        decltype(std::make_integer_sequence<
                 rank_type, ViewType::extents_type::rank()>{})>>::type;

    template <typename... Args, rank_type N>
    auto get_index(std::integral_constant<rank_type, N>, Args&&... idxs) const {
        if constexpr (((N == Indices) || ...)) {
            return zipper::full_extent;
        } else {
            constexpr static rank_type Index =
                traits::index_remover::full_rank_to_reduced_indices[N];
            static_assert(Index <= sizeof...(Args));
            return zipper::detail::pack_index<Index>(
                std::forward<Args>(idxs)...);
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
        // spdlog::info("Computing stuff! {}: {} {} {}", diag.extents(),
        // diag(0),
        //              diag(1), diag(2));
        return reductions::CoefficientSum(diag)();
    }

    template <typename... Args>
    value_type coeff(Args&&... idxs) const {
        zipper::detail::extents::indices_in_range(extents(), idxs...);
        if constexpr (sizeof...(Indices) == 0) {
            return view().coeff(std::forward<Args>(idxs)...);
        } else {
            return _coeff(
                std::make_integer_sequence<rank_type,
                                           ViewType::extents_type::rank()>{},
                std::forward<Args>(idxs)...);
        }
    }

};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

}  // namespace unary
}  // namespace zipper::views
#endif
