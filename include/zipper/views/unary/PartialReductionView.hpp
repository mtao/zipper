#if !defined(ZIPPER_VIEWS_UNARY_PARTIALREDUCTIONVIEW_HPP)
#define ZIPPER_VIEWS_UNARY_PARTIALREDUCTIONVIEW_HPP

#include "UnaryViewBase.hpp"
#include "detail/invert_integer_sequence.hpp"
#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/detail/extents/static_extents_to_array.hpp"
#include "zipper/detail/pack_index.hpp"
#include "zipper/utils/extents/extents_formatter.hpp"
#include "zipper/views/concepts/ReductionViewLike.hpp"
#include "zipper/views/reductions/CoefficientSum.hpp"

namespace zipper::views {
namespace unary {
template <zipper::concepts::QualifiedViewDerived ViewType,
          // template <zipper::concepts::QualifiedViewDerived> typename ReductionView,
          template <typename> typename ReductionView, rank_type... Indices>
// requires ReductionView<ViewType>
class PartialReductionView;

}
template <zipper::concepts::QualifiedViewDerived ViewType,
          // template <zipper::concepts::QualifiedViewDerived> typename ReductionView,
          template <typename> typename ReductionView, rank_type... Indices>
struct detail::ViewTraits<
    unary::PartialReductionView<ViewType, ReductionView, Indices...>>
    : public zipper::views::unary::detail::DefaultUnaryViewTraits<ViewType,
                                                                  true> {
    using Base = detail::ViewTraits<std::decay_t<ViewType>>;
    using child_extents_type = Base::extents_type;
    using index_remover =
        unary::detail::invert_integer_sequence<Base::extents_type::rank(),
                                               Indices...>;
    using extents_type = typename index_remover::template assign_types<
        zipper::extents, zipper::detail::extents::static_extents_to_array_v<
                             typename Base::extents_type>>;
    //using reduced_extents_type = zipper::extents<Indices...>;
    using value_type = Base::value_type;
    constexpr static bool is_writable = false;
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
};

namespace unary {
// indices are the indices being traced
template <zipper::concepts::QualifiedViewDerived ViewType,
          // template <zipper::concepts::QualifiedViewDerived> typename ReductionView,
          template <typename> typename ReductionView, rank_type... Indices>
// requires ReductionView<ViewType>
class PartialReductionView
    : public UnaryViewBase<
          PartialReductionView<ViewType, ReductionView, Indices...>, ViewType> {
   public:
    using self_type = PartialReductionView<ViewType, ReductionView, Indices...>;
    using traits = zipper::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;
    using Base = UnaryViewBase<self_type, ViewType>;
    using Base::extent;
    using Base::view;
    using child_extents_type = traits::child_extents_type;

    PartialReductionView(ViewType& b)
        : Base(b, traits::index_remover::get_extents(b.extents())) {}
    PartialReductionView() = delete;
    PartialReductionView& operator=(const PartialReductionView&) = delete;
    PartialReductionView& operator=(PartialReductionView&&) = delete;
    PartialReductionView(PartialReductionView&& o) = default;
    PartialReductionView(const PartialReductionView& o) = default;
    // PartialReductionView(const PartialReductionView& o) = default;//:
    // Base(o.view(),traits::index_remover::get_extents(o.extents())){ }
    // PartialReductionView(PartialReductionView&& o):
    // PartialReductionView(o.view()) { }
    //: Base(b), m_extents(swizzler_type::swizzle_extents(b.extents())) {}

    template <typename>
    struct slice_type_;
    template <rank_type... N>
    struct slice_type_<std::integer_sequence<rank_type, N...>> {
        using type =
            SliceView<const ViewType,
                      std::conditional_t<traits::index_remover::in_sequence(N),
                                         rank_type, full_extent_t>...>;
    };

    using slice_type = slice_type_<std::decay_t<
        decltype(std::make_integer_sequence<
                 rank_type, child_extents_type::rank()>{})>>::type;

    template <typename... Args, rank_type N>
    auto get_index(std::integral_constant<rank_type, N>, Args&&... idxs) const {
        if constexpr (((N == Indices) || ...)) {
            return zipper::full_extent_t{};
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
        requires(sizeof...(N) == child_extents_type::rank())
    {
        const auto slice =
            slice_type(view(), get_index(std::integral_constant<rank_type, N>{},
                                         std::forward<Args>(idxs)...)...);

        ReductionView<const slice_type> v(slice);
        value_type val = v();
        return val;
    }

    template <typename... Args>
    value_type coeff(Args&&... idxs) const {
        zipper::utils::extents::indices_in_range(extents(), idxs...);
        return _coeff(
            std::make_integer_sequence<rank_type,
                                       child_extents_type::rank()>{},
            std::forward<Args>(idxs)...);
    }

};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

}  // namespace unary
}  // namespace zipper::views
#endif
