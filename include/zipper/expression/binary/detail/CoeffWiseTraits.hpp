
#if !defined(ZIPPER_EXPRESSIONS_BINARY_DETAIL_COEFFWISETRAITS_HPP)
#define ZIPPER_EXPRESSIONS_BINARY_DETAIL_COEFFWISETRAITS_HPP
#include "zipper/expression/detail/ExpressionTraits.hpp"

namespace zipper::expression::binary::detail {

template <typename, typename> struct coeffwise_extents_values;

/// Same-rank merge: static extents win over dynamic extents.
template <index_type... Idxs, index_type... Idxs2>
    requires(sizeof...(Idxs) == sizeof...(Idxs2))
struct coeffwise_extents_values<extents<Idxs...>, extents<Idxs2...>> {
  using merged_extents_type =
      extents<(Idxs == std::dynamic_extent ? Idxs2 : Idxs)...>;
};

/// Rank-0 (infinite) on LHS: result takes the RHS extents.
template <index_type... Idxs>
    requires(sizeof...(Idxs) > 0)
struct coeffwise_extents_values<extents<>, extents<Idxs...>> {
  using merged_extents_type = extents<Idxs...>;
};

/// Rank-0 (infinite) on RHS: result takes the LHS extents.
template <index_type... Idxs>
    requires(sizeof...(Idxs) > 0)
struct coeffwise_extents_values<extents<Idxs...>, extents<>> {
  using merged_extents_type = extents<Idxs...>;
};

/// Both rank-0 (infinite): result is rank-0.
template <>
struct coeffwise_extents_values<extents<>, extents<>> {
  using merged_extents_type = extents<>;
};

template <typename A, typename B> struct CoeffWiseTraits {
  using LHSTraits = zipper::expression::detail::ExpressionTraits<A>;
  using RHSTraits = zipper::expression::detail::ExpressionTraits<B>;
  // static_assert(LHSTraits::value_type == RHSTraits::value_type);
  using value_type = typename LHSTraits::value_type;
  using extents_type = // typename LHSTraits::extents_type;
      coeffwise_extents_values<
          typename LHSTraits::extents_type,
          typename RHSTraits::extents_type>::merged_extents_type;
};
} // namespace zipper::expression::binary::detail
#endif
