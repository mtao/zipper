
#if !defined(ZIPPER_EXPRESSION_UNARY_REPEAT_HPP)
#define ZIPPER_EXPRESSION_UNARY_REPEAT_HPP

#include "UnaryExpressionBase.hpp"
#include "zipper/concepts/Extents.hpp"
#include "zipper/expression/binary/TensorProduct.hpp"

namespace zipper::expression {
namespace unary {
enum class RepeatMode { Left, Right };
template <RepeatMode Mode, rank_type Count,
          zipper::concepts::QualifiedExpression Child>
class Repeat;

namespace _detail_repeat {
template <RepeatMode Mode, rank_type Count, zipper::concepts::Extents ET>
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
    static_assert(zipper::concepts::Extents<left_extents_type>);
    static_assert(zipper::concepts::Extents<right_extents_type>);
    using tensor_type =
        typename binary::_detail_tensor::tensor_coeffwise_extents_values<
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
}  // namespace _detail_repeat

}  // namespace unary
template <unary::RepeatMode Mode, rank_type Count,
          zipper::concepts::QualifiedExpression Child>
struct detail::ExpressionTraits<unary::Repeat<Mode, Count, Child>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<Child, true> {
    using BaseTraits = expression::detail::ExpressionTraits<Child>;
    using base_extents_type = typename BaseTraits::extents_type;
    constexpr static bool is_writable = false;
    constexpr static bool is_coefficient_consistent = true;
    constexpr static bool is_value_based = false;
    // Repeat re-indexes but has no coeff_ref — not referrable or assignable
    constexpr static zipper::detail::AccessFeatures access_features = {
        .is_const = true,
        .is_reference = false,
        .is_alias_free = BaseTraits::access_features.is_alias_free,
    };
    consteval static auto is_const_valued() -> bool {
      return access_features.is_const;
    }
    consteval static auto is_reference_valued() -> bool {
      return access_features.is_reference;
    }
    consteval static auto is_assignable() -> bool {
      return !is_const_valued() && is_reference_valued();
    }
    consteval static auto is_referrable() -> bool {
      return access_features.is_reference;
    }
    using helper_type =
        unary::_detail_repeat::RepeatHelper<Mode, Count,
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
          zipper::concepts::QualifiedExpression Child>
class Repeat : public UnaryExpressionBase<Repeat<Mode, Count, Child>, Child> {
    //

   public:
    using self_type = Repeat<Mode, Count, Child>;
    using Base = UnaryExpressionBase<Repeat<Mode, Count, Child>, Child>;
    using traits = zipper::expression::detail::ExpressionTraits<self_type>;
    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    constexpr static bool is_static = extents_traits::is_static;
    using value_type = traits::value_type;
    constexpr static rank_type base_rank = traits::base_rank;
    constexpr static rank_type pad_offset = traits::offset_rank;

    Repeat(const Child& a)
        requires(is_static)
        : Base(a) {}
    Repeat(const Child& a)
        requires(!is_static)
        : Base(a, traits::make_extents(a.extents())) {}
    using Base::expression;
    template <typename... Args, rank_type... ranks>
    auto get_value(std::integer_sequence<rank_type, ranks...>,
                   Args&&... args) const -> decltype(auto) {
        return expression()(zipper::detail::pack_index<ranks + pad_offset>(
            std::forward<Args>(args)...)...);
    }
    template <typename... Args>
    value_type coeff(Args&&... args) const {
        return get_value(std::make_integer_sequence<rank_type, base_rank>{},
                         std::forward<Args>(args)...);
    }
};

}  // namespace unary
}  // namespace zipper::expression

#endif
