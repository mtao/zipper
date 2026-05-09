#if !defined(ZIPPER_FORMBASE_HXX)
#define ZIPPER_FORMBASE_HXX

#include "FormBase.hpp"
#include "TensorBase.hxx"
#include "expression/nullary/MDSpan.hpp"
#include "zipper/detail/declare_operations.hpp"
#include "zipper/expression/binary/ArithmeticExpressions.hpp"
#include "zipper/expression/binary/FormTensorProduct.hpp"
#include "zipper/expression/binary/WedgeProduct.hpp"
#include "zipper/expression/binary/ZeroAwareOperation.hpp"
#include "zipper/expression/unary/ScalarArithmetic.hpp"

namespace zipper {

// Deduction guide from std::mdspan
template <typename T, typename Extents, typename Layout, typename Accessor>
FormBase(zipper::mdspan<T, Extents, Layout, Accessor>)
    -> FormBase<expression::nullary::MDSpan<T, Extents, Layout, Accessor>>;

UNARY_DECLARATION(FormBase, LogicalNot, operator!)
UNARY_DECLARATION(FormBase, BitNot, operator~)
UNARY_DECLARATION(FormBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(FormBase, Plus, operator+)
SCALAR_BINARY_DECLARATION(FormBase, Minus, operator-)
SCALAR_BINARY_DECLARATION(FormBase, Multiplies, operator*)
SCALAR_BINARY_DECLARATION(FormBase, Divides, operator/)

ZERO_AWARE_BINARY_DECLARATION(FormBase, Plus, operator+)
ZERO_AWARE_BINARY_DECLARATION(FormBase, Minus, operator-)

template <typename Expr1, typename Expr2>
    requires(concepts::Form<std::decay_t<Expr1>>
             && concepts::Form<std::decay_t<Expr2>>)
auto operator^(Expr1 &&lhs, Expr2 &&rhs) {
    using A = detail::forwarded_expression_t<Expr1>;
    using B = detail::forwarded_expression_t<Expr2>;
    using V = expression::binary::WedgeProduct<A, B>;
    return FormBase<V>(std::in_place,
                       std::forward<Expr1>(lhs).expression(),
                       std::forward<Expr2>(rhs).expression());
}

template <typename Expr1, typename Expr2>
    requires(concepts::Form<std::decay_t<Expr1>>
             && concepts::Tensor<std::decay_t<Expr2>>)
auto operator*(Expr1 &&lhs, Expr2 &&rhs) {
    using A = detail::forwarded_expression_t<Expr1>;
    using B = detail::forwarded_expression_t<Expr2>;
    using V = expression::binary::FormTensorProduct<A, B>;

    if constexpr (V::result_is_form) {
        return FormBase<V>(std::in_place,
                           std::forward<Expr1>(lhs).expression(),
                           std::forward<Expr2>(rhs).expression());
    } else {
        return TensorBase<V>(std::in_place,
                             std::forward<Expr1>(lhs).expression(),
                             std::forward<Expr2>(rhs).expression());
    }
}
template <typename Expr1, typename Expr2>
    requires(concepts::Form<std::decay_t<Expr1>>
             && concepts::Vector<std::decay_t<Expr2>>)
auto operator*(Expr1 &&lhs, Expr2 &&rhs) {
    using A = detail::forwarded_expression_t<Expr1>;
    using B = detail::forwarded_expression_t<Expr2>;
    using V = expression::binary::FormTensorProduct<A, B>;
    if constexpr (V::result_is_form) {
        return FormBase<V>(std::in_place,
                           std::forward<Expr1>(lhs).expression(),
                           std::forward<Expr2>(rhs).expression());
    } else {
        return TensorBase<V>(std::in_place,
                             std::forward<Expr1>(lhs).expression(),
                             std::forward<Expr2>(rhs).expression());
    }
}

/// @brief Form × Matrix → FormBase.
///
/// Computes the contraction of a rank-1 form with a rank-2 matrix, producing a
/// rank-1 form.  Mathematically: (f * M)(j) = sum_i f(i) * M(i, j).
///
/// The underlying FormTensorProduct expression computes the correct result but
/// marks `result_is_form = false` because a_rank < b_rank.  This overload
/// unconditionally wraps the result in FormBase, which is the correct semantic
/// type for a row vector produced by contracting a covector with a matrix.
template <typename Expr1, typename Expr2>
    requires(concepts::Form<std::decay_t<Expr1>>
             && concepts::Matrix<std::decay_t<Expr2>>)
auto operator*(Expr1 &&lhs, Expr2 &&rhs) {
    using A = detail::forwarded_expression_t<Expr1>;
    using B = detail::forwarded_expression_t<Expr2>;
    using V = expression::binary::FormTensorProduct<A, B>;

    // The result is always rank-1 (one index contracted out of two), so it is
    // semantically a form regardless of the result_is_form flag.
    return FormBase<V>(std::in_place,
                       std::forward<Expr1>(lhs).expression(),
                       std::forward<Expr2>(rhs).expression());
}

} // namespace zipper

#endif
