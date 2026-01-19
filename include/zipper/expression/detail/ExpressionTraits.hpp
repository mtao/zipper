#if !defined(ZIPPER_EXPRESSION_DETAIL_EXPRESSIONTRAITS_HPP)
#define ZIPPER_EXPRESSION_DETAIL_EXPRESSIONTRAITS_HPP
#include "zipper/types.hpp"

namespace zipper::expression::detail {
template <typename T> struct ExpressionTraits;

template <typename T>
struct ExpressionTraits<const T> : public ExpressionTraits<T> {
  constexpr static bool is_const = true;
};

template <typename T>
struct ExpressionTraits<T &> : public ExpressionTraits<T> {};
// NOTE: template parameters should NOT be used in this struct so that derived
// can overwrite them
template <typename ValueType = void, typename Extents = zipper::dextents<0>>
struct DefaultExpressionTraits {
  using value_type = std::remove_const_t<ValueType>;
  using extents_type = Extents;

  constexpr static bool is_const = std::is_const_v<ValueType>;
  constexpr static bool is_assignable = false;

  constexpr static bool is_resizable = false;

  // guarantees that V(j) = f(...) cannot depend on V(k) for j != k)
  constexpr static bool is_alias_free = false;

  // returns true if a dimension is sparse, false if dense
  consteval static auto is_sparse(rank_type) -> bool { return false; }
};
//{
//    using value_type = ...;
//    using extents_type = ...;
//    using layout_policy = default_layout_policy;
//    using accessor_policy = default_accessor_policy<T>;
//};

} // namespace zipper::expression::detail
#endif
