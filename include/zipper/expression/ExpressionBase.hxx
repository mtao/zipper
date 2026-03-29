#if !defined(ZIPPER_expression_EXPRESSIONBASE_HXX)
#define ZIPPER_expression_EXPRESSIONBASE_HXX

#include "ExpressionBase.hpp"
#include "zipper/detail/assert.hpp"
#include "zipper/utils/extents/indices_in_range.hpp"
#include <functional>

namespace zipper::expression {

template <typename Derived>
template <zipper::concepts::Index... Indices>
auto ExpressionBase<Derived>::coeff(Indices &&...indices) const -> value_type
  requires(rank == 0 || sizeof...(Indices) == rank)
{
  return derived().coeff(std::forward<Indices>(indices)...);
}
template <typename Derived>
template <zipper::concepts::Index... Indices>
auto ExpressionBase<Derived>::coeff_ref(Indices &&...indices) -> value_type &
  requires(traits::is_referrable() && !traits::is_const_valued() &&
           (rank == 0 || sizeof...(Indices) == rank))
{
  return derived().coeff_ref(std::forward<Indices>(indices)...);
}

template <typename Derived>
template <zipper::concepts::Index... Indices>
auto ExpressionBase<Derived>::const_coeff_ref(Indices &&...indices) const
    -> const value_type &
  requires(traits::is_referrable() &&
           (rank == 0 || sizeof...(Indices) == rank))
{
  return derived().const_coeff_ref(std::forward<Indices>(indices)...);
}

namespace unary {
template <zipper::concepts::QualifiedExpression ExpressionType,
          typename... Slices>
class Slice;
}
template <typename Derived>
template <zipper::concepts::IndexSlice... Slices>
auto ExpressionBase<Derived>::const_access_slice(Slices &&...slices) const {
  return unary::Slice<const Derived&, std::decay_t<Slices>...>(
      derived(), std::forward<Slices>(slices)...);
}

template <typename Derived>
template <zipper::concepts::IndexSlice... Slices>
auto ExpressionBase<Derived>::access_slice(Slices &&...slices) {
  return unary::Slice<Derived&, std::decay_t<Slices>...>(
      derived(), std::forward<Slices>(slices)...);
}

template <typename Derived>
template <zipper::concepts::IndexArgument... Args>
auto ExpressionBase<Derived>::operator()(Args &&...idxs) const -> decltype(auto)

{
  return const_access_pack(std::forward<Args>(idxs)...);
}
template <typename Derived>
template <zipper::concepts::IndexArgument... Args>
auto ExpressionBase<Derived>::operator()(Args &&...idxs) -> decltype(auto)
  requires(is_assignable && !is_const_valued)

{
  return access_pack(std::forward<Args>(idxs)...);
}

template <typename Derived>
template <zipper::concepts::IndexArgument... Args>
auto ExpressionBase<Derived>::const_access_pack(Args &&...idxs) const
    -> decltype(auto)

{
  if constexpr (zipper::concepts::IndexPack<Args...>) {
    return const_access_index_pack(std::forward<Args>(idxs)...);
  } else {
    return const_access_slice(std::forward<Args>(idxs)...);
  }
}

template <typename Derived>
template <zipper::concepts::Index... Args>
auto ExpressionBase<Derived>::access_index_pack(Args &&...idxs)
    -> decltype(auto) {
  ZIPPER_ASSERT(zipper::utils::extents::indices_in_range(extents(), idxs...));
  if constexpr (traits::is_referrable()) {
    return coeff_ref(std::forward<Args>(idxs)...);
  } else {
    return coeff(std::forward<Args>(idxs)...);
  }
}

template <typename Derived>
template <zipper::concepts::Index... Args>
auto ExpressionBase<Derived>::const_access_index_pack(Args &&...idxs) const
    -> decltype(auto) {
  ZIPPER_ASSERT(zipper::utils::extents::indices_in_range(extents(), idxs...));
  if constexpr (traits::is_referrable() && !traits::has_known_zeros) {
    return const_coeff_ref(std::forward<Args>(idxs)...);
  } else {
    return coeff(std::forward<Args>(idxs)...);
  }
}

template <typename Derived>
template <zipper::concepts::IndexArgument... Args>
auto ExpressionBase<Derived>::access_pack(Args &&...idxs) -> decltype(auto)

{
  if constexpr (zipper::concepts::IndexPack<Args...>) {
    return access_index_pack(std::forward<Args>(idxs)...);
  } else {
    return access_slice(std::forward<Args>(idxs)...);
  }
}

template <typename Derived>
template <zipper::concepts::IndexArgumentPackTuple Tuple>
auto ExpressionBase<Derived>::access_tuple(const Tuple &t) -> decltype(auto) {
  return std::apply(std::bind_front(&ExpressionBase::access_pack, this), t);
}
template <typename Derived>
template <zipper::concepts::IndexArgumentPackTuple Tuple>
auto ExpressionBase<Derived>::const_access_tuple(const Tuple &t) const
    -> decltype(auto) {
  return std::apply(std::bind_front(&ExpressionBase::const_access_pack, this),
                    t);
}

} // namespace zipper::expression
#endif
