

#if !defined(ZIPPER_EXPRESSIONS_BINARY_OPERATIONVIEW_HPP)
#define ZIPPER_EXPRESSIONS_BINARY_OPERATIONVIEW_HPP

#include "BinaryExpressionBase.hpp"
#include "detail/CoeffWiseTraits.hpp"
#include "zipper/concepts/Expression.hpp"
#include "zipper/utils/extents/extents_formatter.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winline"
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wswitch-enum"
#if !defined(__clang__)
#pragma GCC diagnostic ignored "-Wabi-tag"
#endif
#include <fmt/format.h>
#pragma GCC diagnostic pop
#if defined(ZIPPER_FMT_OVERRIDES_DISABLED)
#include "zipper/utils/extents/as_array.hpp"
#endif

#include "zipper/utils/extents/extents_formatter.hpp"
namespace zipper::expression {
namespace binary {
template <zipper::concepts::QualifiedExpression A,
          zipper::concepts::QualifiedExpression B, typename Op>
class Operation;

}
template <concepts::QualifiedExpression A, concepts::QualifiedExpression B, typename Op>
struct detail::ExpressionTraits<binary::Operation<A, B, Op>>
    : public binary::detail::DefaultBinaryExpressionTraits<A, B> {
  using ATraits = detail::ExpressionTraits<A>;
  using BTraits = detail::ExpressionTraits<B>;
  using ConvertExtentsUtil =
      binary::detail::coeffwise_extents_values<typename ATraits::extents_type,
                                               typename BTraits::extents_type>;
  using extents_type = typename ConvertExtentsUtil::merged_extents_type;
  using value_type = decltype(std::declval<Op>()(
      std::declval<typename ATraits::value_type>(),
      std::declval<typename BTraits::value_type>()));
};

namespace binary {
template <zipper::concepts::QualifiedExpression A,
          zipper::concepts::QualifiedExpression B, typename Op>
class Operation
    : public BinaryExpressionBase<Operation<A, B, Op>, const A,
                                  const B> {
public:
  using self_type = Operation<A, B, Op>;
  using traits = zipper::expression::detail::ExpressionTraits<self_type>;
  using extents_type = typename traits::extents_type;
  using extents_traits = zipper::detail::ExtentsTraits<extents_type>;
  using Base = BinaryExpressionBase<self_type, const A, const B>;
  using value_type = traits::value_type;
  constexpr static bool is_static = extents_traits::is_static;

  using a_extents_type = traits::ATraits::extents_type;
  using b_extents_type = traits::BTraits::extents_type;

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
  Operation(const A &a, const B &b, const Op &op = {})
      : Base(a, b), m_op(op) {
    if (!valid_input_extents(a.extents(), b.extents())) {
      throw std::runtime_error(
          fmt::format("Operation between {} and {} is invalid",
#if defined(ZIPPER_FMT_OVERRIDES_DISABLED)
                      zipper::utils::extents::as_array(a.extents()),
                      zipper::utils::extents::as_array(b.extents())));
#else
                      a.extents(), b.extents()));
#endif
    }
  }

  value_type get_value(const auto &a, const auto &b) const {
    return m_op(a, b);
  }

  constexpr auto extent(rank_type i) const -> index_type {
    return lhs().extent(i);
  }

  constexpr auto extents() const -> extents_type {
    return extents_traits::make_extents_from(*this);
  }

private:
  Op m_op;
};
template <zipper::concepts::QualifiedExpression A,
          zipper::concepts::QualifiedExpression B, typename Op>
Operation(const A &a, const B &b, const Op &op)
    -> Operation<A, B, Op>;

} // namespace binary
} // namespace zipper::expression
#endif
