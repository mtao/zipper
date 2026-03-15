#if !defined(ZIPPER_AS_HPP)
#define ZIPPER_AS_HPP
#include "concepts/Zipper.hpp"
#include "expression/concepts/capabilities.hpp"
#include <utility> // std::in_place

namespace zipper {

// Forward declarations of Base types used by AS_IMPL
template <concepts::Expression T> class ArrayBase;
template <concepts::Expression T> class ContainerBase;
template <concepts::Expression T> class VectorBase;
template <concepts::Expression T> class MatrixBase;
template <concepts::Expression T> class FormBase;
template <concepts::Expression T> class TensorBase;

// As functions for each basic type

#define AS_IMPL(NAME_LOWER, NAME_UPPER)                                        \
  template <concepts::Zipper ZipperDerived>                         \
  auto as_##NAME_LOWER(ZipperDerived &v) {                                     \
    using Expr = typename ZipperDerived::expression_type;                      \
    constexpr static bool make_const =                                         \
        !expression::concepts::WritableExpression<Expr> ||                     \
        std::is_const_v<ZipperDerived>;                                        \
    using ExprC = std::conditional_t<make_const, const Expr, Expr>;            \
    return NAME_UPPER##Base<ExprC &>(std::in_place, v.expression());            \
  }                                                                            \
  template <concepts::Zipper ZipperDerived>                         \
  auto as_##NAME_LOWER(const ZipperDerived &v) {                               \
    using Expr = typename ZipperDerived::expression_type;                      \
    return NAME_UPPER##Base<const Expr &>(std::in_place, v.expression());      \
  }

AS_IMPL(array, Array)
AS_IMPL(container, Container)
AS_IMPL(vector, Vector)
AS_IMPL(matrix, Matrix)
AS_IMPL(form, Form)
AS_IMPL(tensor, Tensor)

} // namespace zipper
#endif
