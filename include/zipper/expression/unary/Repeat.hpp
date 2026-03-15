
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
        return extents<((void)I, dynamic_extent)...>{};
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

/// Implementation details for Repeat expressions.
///
/// Holds the child's base rank and the offset rank (number of dimensions
/// prepended or appended). The Repeat class body needs these to correctly
/// map output indices to child indices.
template <unary::RepeatMode Mode, rank_type Count,
          zipper::concepts::QualifiedExpression Child>
struct detail::ExpressionDetail<unary::Repeat<Mode, Count, Child>> {
    using BaseTraits = expression::detail::ExpressionTraits<std::decay_t<Child>>;
    using helper_type =
        unary::_detail_repeat::RepeatHelper<Mode, Count,
                                    typename BaseTraits::extents_type>;
    constexpr static rank_type base_rank = helper_type::base_rank;
    constexpr static rank_type offset_rank = helper_type::offset_rank;
};

template <unary::RepeatMode Mode, rank_type Count,
          zipper::concepts::QualifiedExpression Child>
struct detail::ExpressionTraits<unary::Repeat<Mode, Count, Child>>
    : public zipper::expression::unary::detail::DefaultUnaryExpressionTraits<
          Child,
          zipper::detail::AccessFeatures{.is_const = true,
                                         .is_reference = false}> {
    using _Detail = detail::ExpressionDetail<unary::Repeat<Mode, Count, Child>>;
    constexpr static bool is_coefficient_consistent = true;
    constexpr static bool is_value_based = false;
    using extents_type = typename _Detail::helper_type::extents_type;
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
    using detail_type = zipper::expression::detail::ExpressionDetail<self_type>;
    using extents_type = traits::extents_type;
    using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
    constexpr static bool is_static = extents_traits::is_static;
    using value_type = traits::value_type;
    constexpr static rank_type base_rank = detail_type::base_rank;
    constexpr static rank_type pad_offset = detail_type::offset_rank;

    template <typename U>
      requires std::constructible_from<typename Base::storage_type, U &&>
    Repeat(U &&a)
        : Base(std::forward<U>(a)) {}

    /// Recursively deep-copy child so the result owns all data.
    auto make_owned() const {
        auto owned_child = expression().make_owned();
        return Repeat<Mode, Count, const decltype(owned_child)>(
            std::move(owned_child));
    }

    constexpr auto extent(rank_type i) const -> index_type {
        if constexpr (Mode == RepeatMode::Left) {
            if (i < Count) {
                return extents_type::static_extent(i);  // dummy dim (0)
            } else {
                return expression().extent(i - Count);
            }
        } else {
            if (i < base_rank) {
                return expression().extent(i);
            } else {
                return extents_type::static_extent(i);  // dummy dim (0)
            }
        }
    }

    constexpr auto extents() const -> extents_type {
        return extents_traits::make_extents_from(*this);
    }

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
