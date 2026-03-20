

#if !defined(ZIPPER_EXPRESSIONS_BINARY_OPERATIONVIEW_HPP)
#define ZIPPER_EXPRESSIONS_BINARY_OPERATIONVIEW_HPP

#include "BinaryExpressionBase.hpp"
#include "detail/CoeffWiseTraits.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/utils/extents/extents_formatter.hpp"

#include <format>

namespace zipper::expression {
namespace binary {
template <zipper::concepts::QualifiedExpression A,
          zipper::concepts::QualifiedExpression B, typename Op>
class Operation;

}

/// Implementation details for Operation expressions.
///
/// Holds child traits aliases and the coefficient-wise extents merge utility
/// used by both the traits specialization (to compute extents_type/value_type)
/// and the class body (to access child extents types for validation).
template <zipper::concepts::QualifiedExpression A,
          zipper::concepts::QualifiedExpression B, typename Op>
struct detail::ExpressionDetail<binary::Operation<A, B, Op>>
    : public binary::detail::DefaultBinaryExpressionDetail<A, B> {
  using _Base = binary::detail::DefaultBinaryExpressionDetail<A, B>;
  using ATraits = typename _Base::ATraits;
  using BTraits = typename _Base::BTraits;
  using ConvertExtentsUtil =
      binary::detail::coeffwise_extents_values<typename ATraits::extents_type,
                                               typename BTraits::extents_type>;
};

template <zipper::concepts::QualifiedExpression A, zipper::concepts::QualifiedExpression B, typename Op>
struct detail::ExpressionTraits<binary::Operation<A, B, Op>>
    : public binary::detail::DefaultBinaryExpressionTraits<A, B> {
  using _Detail = detail::ExpressionDetail<binary::Operation<A, B, Op>>;
  using extents_type = typename _Detail::ConvertExtentsUtil::merged_extents_type;
  using value_type = decltype(std::declval<Op>()(
      std::declval<typename _Detail::ATraits::value_type>(),
      std::declval<typename _Detail::BTraits::value_type>()));
};

namespace binary {
template <zipper::concepts::QualifiedExpression A,
          zipper::concepts::QualifiedExpression B, typename Op>
class Operation
    : public BinaryExpressionBase<Operation<A, B, Op>, A, B> {
public:
  using self_type = Operation<A, B, Op>;
  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using detail_type = zipper::expression::detail::ExpressionDetail<self_type>;
  using extents_type = typename traits::extents_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  using Base = BinaryExpressionBase<self_type, A, B>;
  using value_type = traits::value_type;
  constexpr static bool is_static = extents_traits::is_static;

  using a_extents_type = typename detail_type::ATraits::extents_type;
  using b_extents_type = typename detail_type::BTraits::extents_type;

  constexpr static bool valid_input_extents(const a_extents_type &a,
                                            const b_extents_type &b) {
    if constexpr (a_extents_type::rank() == 0 ||
                  b_extents_type::rank() == 0) { // infinite tensor case
      return true;
    } else if constexpr (a_extents_type::rank() == b_extents_type::rank()) {
      for (rank_type j = 0; j < a_extents_type::rank(); ++j) {
        // dynamic extent are infinite dimensions cases
        if (a.extent(j) != b.extent(j) && a.extent(j) != std::dynamic_extent &&
            b.extent(j) != std::dynamic_extent) {
          return false;
        }
      }
      return true;

    } else {
      return false;
    }
  }

  using Base::lhs;
  using Base::rhs;

  template <typename U, typename V>
    requires std::constructible_from<typename Base::lhs_storage_type, U&&> &&
             std::constructible_from<typename Base::rhs_storage_type, V&&>
  Operation(U&& a, V&& b, const Op &op = {})
      : Base(std::forward<U>(a), std::forward<V>(b)), m_op(op) {
    if (!valid_input_extents(lhs().extents(), rhs().extents())) {
      throw std::runtime_error(
          std::format("Operation between {} and {} is invalid",
                      lhs().extents(), rhs().extents()));
    }
  }

  value_type get_value(const auto &a, const auto &b) const {
    return m_op(a, b);
  }

  constexpr auto extent(rank_type i) const -> index_type {
    if constexpr (a_extents_type::rank() == 0) {
      return rhs().extent(i);
    } else {
      return lhs().extent(i);
    }
  }

  constexpr auto extents() const -> extents_type {
    return extents_traits::make_extents_from(*this);
  }

  /// Recursively deep-copy children so the result owns all data.
  auto make_owned() const {
      auto owned_a = lhs().make_owned();
      auto owned_b = rhs().make_owned();
      return Operation<decltype(owned_a), decltype(owned_b), Op>(
          std::move(owned_a), std::move(owned_b), m_op);
  }

private:
  Op m_op;
};
template <zipper::concepts::Expression A,
          zipper::concepts::Expression B, typename Op>
Operation(const A &a, const B &b, const Op &op)
    -> Operation<const A&, const B&, Op>;

} // namespace binary
} // namespace zipper::expression
#endif
