
#if !defined(ZIPPER_VIEWS_UNARY_REPEATVIEW_HPP)
#define ZIPPER_VIEWS_UNARY_REPEATVIEW_HPP

#include "UnaryViewBase.hpp"
#include "zipper/concepts/ExtentsType.hpp"
#include "zipper/views/binary/TensorProductView.hpp"

namespace zipper::views {
namespace unary {
enum class RepeatMode { Left, Right };
template <RepeatMode Mode, rank_type Count,
          zipper::concepts::QualifiedViewDerived Child>
class RepeatView;

namespace detail {
template <RepeatMode Mode, rank_type Count, zipper::concepts::ExtentsType ET>
struct RepeatHelper {
    template <size_t... I>
    constexpr static auto make_repeat_extents_type(
        std::integer_sequence<size_t, I...>) {
        return extents<(I - I)...>{};
        //
    }
    using repeat_extents_type = std::decay_t<decltype(make_repeat_extents_type(
        std::make_index_sequence<Count>{}))>;

    using left_extents_type =
        std::conditional_t<Mode == RepeatMode::Left, repeat_extents_type, ET>;
    using right_extents_type =
        std::conditional_t<Mode == RepeatMode::Right, repeat_extents_type, ET>;
    static_assert(zipper::concepts::ExtentsType<left_extents_type>);
    static_assert(zipper::concepts::ExtentsType<right_extents_type>);
    using tensor_type =
        typename views::binary::detail::tensor_coeffwise_extents_values<
            left_extents_type, right_extents_type>;

    using extents_type = typename tensor_type::product_extents_type;
    constexpr static rank_type base_rank = ET::rank();
    constexpr static rank_type offset_rank =
        Mode == RepeatMode::Left ? Count : 0;
    constexpr static extents_type make_extents(const ET& e) {
        if constexpr (Mode == RepeatMode::Left) {
            return tensor_type::merge(repeat_extents_type{}, e);
        } else {
            return tensor_type::merge(e, repeat_extents_type{});
        }
    }
};
}  // namespace detail

}  // namespace unary
template <unary::RepeatMode Mode, rank_type Count,
          zipper::concepts::QualifiedViewDerived Child>
struct detail::ViewTraits<unary::RepeatView<Mode, Count, Child>>
    : public zipper::views::unary::detail::DefaultUnaryViewTraits<Child, true> {
    using BaseTraits = views::detail::ViewTraits<Child>;
    using base_extents_type = typename BaseTraits ::extents_type;
    constexpr static bool is_writable = false;
    constexpr static bool is_coefficient_consistent = true;
    constexpr static bool is_value_based = false;
    using helper_type =
        unary::detail::RepeatHelper<Mode, Count,
                                    typename BaseTraits::extents_type>;
    using extents_type = helper_type::extents_type;
    constexpr static rank_type base_rank = helper_type::base_rank;
    constexpr static rank_type offset_rank = helper_type::offset_rank;

    constexpr static extents_type make_extents(const base_extents_type& e) {
        return helper_type::make_extents(e);
    }
};
namespace unary {
template <RepeatMode Mode, rank_type Count,
          zipper::concepts::QualifiedViewDerived Child>
class RepeatView : public UnaryViewBase<RepeatView<Mode, Count, Child>, Child> {
    //

   public:
    using self_type = RepeatView<Mode, Count, Child>;
    using Base = UnaryViewBase<RepeatView<Mode, Count, Child>, Child>;
    using traits = zipper::views::detail::ViewTraits<self_type>;
    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    constexpr static bool is_static = extents_traits::is_static;
    using value_type = traits::value_type;
    constexpr static rank_type base_rank = traits::base_rank;
    constexpr static rank_type pad_offset = traits::offset_rank;

    RepeatView(const Child& a)
        requires(is_static)
        : Base(a) {}
    RepeatView(const Child& a)
        requires(!is_static)
        : Base(a, traits::make_extents(a.extents())) {}
    using Base::view;
    template <typename... Args, rank_type... ranks>
    auto get_value(std::integer_sequence<rank_type, ranks...>,
                   Args&&... args) const -> decltype(auto) {
        return view()(zipper::detail::pack_index<ranks + pad_offset>(
            std::forward<Args>(args)...)...);
    }
    template <typename... Args>
    value_type coeff(Args&&... args) const {
        return get_value(std::make_integer_sequence<rank_type, base_rank>{},
                         std::forward<Args>(args)...);
    }
};

// template <unary::RepeatMode Mode, rank_type Count,
//           zipper::concepts::QualifiedViewDerived Child>
// RepeatView(const Child& c) -> RepeatView<Mode, Count, Child>;
}  // namespace unary
}  // namespace zipper::views

#endif
