
#if !defined(ZIPPER_expression_unary_DETAIL_COEFFWISETRAITS_HPP)
#define ZIPPER_expression_unary_DETAIL_COEFFWISETRAITS_HPP
#include "zipper/expression/detail/ExpressionTraits.hpp"

namespace zipper::expression::unary::detail {

template <typename A, typename B> struct CoeffWiseTraits {
  using LHSTraits = zipper::expression::detail::ExpressionTraits<A>;
  // static_assert(LHSTraits::value_type == RHSTraits::value_type);
  using value_type = typename LHSTraits::value_type;
  using extents_type = typename LHSTraits::extents_type;
  // coeffwise_extents_values<typename LHSTraits::extents_type, typename
  // RHSTraits::extents_type>::merged_extents_type;
};
} // namespace zipper::expression::unary::detail
#endif
