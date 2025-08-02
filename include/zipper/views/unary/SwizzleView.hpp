

#if !defined(ZIPPER_VIEWS_UNARY_SWIZZLEVIEW_HPP)
#define ZIPPER_VIEWS_UNARY_SWIZZLEVIEW_HPP

#include "UnaryViewBase.hpp"
#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/detail/extents/swizzle_extents.hpp"
#include "zipper/storage/PlainObjectStorage.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"
#include "zipper/views/DimensionedViewBase.hpp"
#include "zipper/views/detail/AssignHelper.hpp"

namespace zipper::views {
namespace unary {
template <zipper::concepts::QualifiedViewDerived ViewType, index_type... Indices>
class SwizzleView;

}
template <zipper::concepts::QualifiedViewDerived QualifiedViewType,
          index_type... Indices>
struct detail::ViewTraits<unary::SwizzleView<QualifiedViewType, Indices...>>
    : public zipper::views::unary::detail::DefaultUnaryViewTraits<
          QualifiedViewType, true> {
    using ViewType = std::decay_t<QualifiedViewType>;
    using swizzler_type = zipper::detail::extents::ExtentsSwizzler<Indices...>;

    // TODO: fix this so ViewTraits are all aware of qulaifiacitons
    using QualifiedBase = zipper::views::detail::ViewTraits<QualifiedViewType>;
    using Base = zipper::views::detail::ViewTraits<QualifiedViewType>;
    using extents_type = swizzler_type::template extents_type_swizzler_t<
        typename Base::extents_type>;
    using value_type = Base::value_type;
    constexpr static bool is_coefficient_consistent = false;
    constexpr static bool is_value_based = false;
};

namespace unary {
template <zipper::concepts::QualifiedViewDerived QualifiedViewType,
          index_type... Indices>
class SwizzleView
    : public UnaryViewBase<SwizzleView<QualifiedViewType, Indices...>,
                           QualifiedViewType> {
   public:
    using self_type = SwizzleView<QualifiedViewType, Indices...>;
    using traits = zipper::views::detail::ViewTraits<self_type>;
    using ViewType = traits::ViewType;
    using extents_type = traits::extents_type;
    using value_type = traits::value_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    using swizzler_type = traits::swizzler_type;
    using Base = UnaryViewBase<self_type, QualifiedViewType>;
    using Base::extent;
    using Base::view;
    constexpr static rank_type internal_rank = ViewType::extents_type::rank();
    constexpr static std::array<rank_type, internal_rank>
        to_internal_rank_indices = swizzler_type::valid_internal_indices;

    SwizzleView(const SwizzleView&) = default;
    SwizzleView(SwizzleView&&) = default;
    SwizzleView& operator=(const SwizzleView&) = delete;
    SwizzleView& operator=(SwizzleView&&) = delete;
    SwizzleView(QualifiedViewType& b)
        : Base(b, swizzler_type::swizzle_extents(b.extents())) {}


    template <typename... Args>
        requires(extents_type::rank() == sizeof...(Args))
    value_type coeff(Args&&... idxs) const {
        return _coeff(swizzler_type::unswizzle(std::forward<Args>(idxs)...),
                      std::make_integer_sequence<rank_type, internal_rank>{});
    }
    template <typename... Args>
    value_type& coeff_ref(Args&&... idxs)
        requires((traits::is_writable) &&
                 (extents_type::rank() == sizeof...(Args)))
    {
        return _coeff_ref(
            swizzler_type::unswizzle(std::forward<Args>(idxs)...),
            std::make_integer_sequence<rank_type, internal_rank>{});
    }
    template <typename... Args>
    const value_type& const_coeff_ref(Args&&... idxs) const
        requires((traits::is_writable) &&
                 (extents_type::rank() == sizeof...(Args)))
    {
        static_assert(extents_type::rank() == sizeof...(Args));
        return _const_coeff_ref(
            swizzler_type::unswizzle(std::forward<Args>(idxs)...),
            std::make_integer_sequence<rank_type, internal_rank>{});
    }

    template <zipper::concepts::ViewDerived V>
    void assign(const V& v)
        requires(
            traits::is_writable &&
            extents_traits::template is_convertable_from<
                typename zipper::views::detail::ViewTraits<V>::extents_type>())
    {
        views::detail::AssignHelper<V, self_type>::assign(v, *this);
    }

   private:
    template <zipper::concepts::TupleLike T, rank_type... ranks>
    auto _coeff(const T& idxs, std::integer_sequence<rank_type, ranks...>) const
        -> value_type {
        return view().coeff(std::get<ranks>(idxs)...);
    }
    template <zipper::concepts::TupleLike T, rank_type... ranks>
    auto _coeff_ref(const T& idxs, std::integer_sequence<rank_type, ranks...>)
        -> value_type&
        requires(traits::is_writable)
    {
        return view().coeff_ref(std::get<ranks>(idxs)...);
    }

    template <zipper::concepts::TupleLike T, rank_type... ranks>
    auto _const_coeff_ref(const T& idxs,
                          std::integer_sequence<rank_type, ranks...>) const
        -> const value_type&
        requires(traits::is_writable)
    {
        return view().const_coeff_ref(std::get<ranks>(idxs)...);
    }
};  // namespace unarytemplate<typenameA,typenameB>class AdditionView

}  // namespace unary
}  // namespace zipper::views
#endif
