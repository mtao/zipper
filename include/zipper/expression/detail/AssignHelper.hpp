#if !defined(ZIPPER_EXPRESSION_DETAIL_ASSIGNHELPER_HPP)
#define ZIPPER_EXPRESSION_DETAIL_ASSIGNHELPER_HPP
#include "zipper/concepts/Expression.hpp"
#include "zipper/detail/ExtentsTraits.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/utils/extents/all_extents_indices.hpp"
#include <tuple>

namespace zipper::expression::nullary {
template <typename ElementType, typename Extents, typename LayoutPolicy,
          typename AccessorPolicy>
class MDArray;
} // namespace zipper::expression::nullary

namespace zipper::expression::detail {

template <zipper::concepts::Expression From, zipper::concepts::Expression To>
struct AssignHelper {
  static_assert(ExpressionTraits<To>::is_writable);
  using to_traits = ExpressionTraits<To>;
  using from_traits = ExpressionTraits<From>;
  using to_extents_type = typename to_traits::extents_type;
  using from_extents_type = typename from_traits::extents_type;

  // "the output sorta traits"
  using traits = to_traits;
  using value_type = typename to_traits::value_type;
  using extents_type = to_extents_type;

  using layout_policy = zipper::default_layout_policy;
  using accessor_policy = zipper::default_accessor_policy<value_type>;
  // using layout_policy = traits::layout_policy;
  // using accessor_policy = traits::accessor_policy;
  // using value_accessor_type = traits::value_accessor_type;

  using to_extents_traits = zipper::detail::ExtentsTraits<to_extents_type>;
  using from_extents_traits = zipper::detail::ExtentsTraits<from_extents_type>;
  static_assert(
      to_extents_traits::template is_convertable_from<from_extents_type>());

  static void assign_direct(const From &from, To &to);

  static void assign(const From &from, To &to);
};

template <zipper::concepts::Expression From, zipper::concepts::Expression To>
void AssignHelper<From, To>::assign_direct(const From &from, To &to) {
  if constexpr (extents_type::rank() == 0) {
    to() = from();
  } else {
    for (const auto &i :
         zipper::utils::extents::all_extents_indices(to.extents())) {
      std::apply(to, i) = std::apply(from, i);
    }
  }
}
template <zipper::concepts::Expression From, zipper::concepts::Expression To>
void AssignHelper<From, To>::assign(const From &from, To &to) {
  using FromTraits = zipper::expression::detail::ExpressionTraits<From>;
  using ToTraits = zipper::expression::detail::ExpressionTraits<To>;
  constexpr static bool assigning_from_infinite =
      FromTraits::extents_type::rank() == 0;
  constexpr static bool should_resize =
      !assigning_from_infinite && ToTraits::is_resizable();
  if constexpr (FromTraits::is_coefficient_consistent) {
    if constexpr (should_resize) {
      to.resize(to_extents_traits::convert_from(from.extents()));
    } else if constexpr (to_extents_traits::is_dynamic &&
                         !assigning_from_infinite) {
      assert(to.extents() == from.extents());
    }

    assign_direct(from, to);
  } else {
    using POS = nullary::MDArray<value_type, extents_type,
                                            layout_policy, accessor_policy>;
    POS pos(to_extents_traits::convert_from(from.extents()));

    AssignHelper<From, POS>::assign_direct(from, pos);
    if constexpr (should_resize) {
      to.resize(to_extents_traits::convert_from(from.extents()));
    }
    AssignHelper<POS, To>::assign_direct(pos, to);
  }
}
} // namespace zipper::expression::detail
#endif
