#if !defined(ZIPPER_VECTORBASE_HXX)
#define ZIPPER_VECTORBASE_HXX

#include "ArrayBase.hxx"
#include "FormBase.hxx"
#include "VectorBase.hpp"
#include "expression/nullary/MDSpan.hpp"
#include "zipper/detail/declare_operations.hpp"
#include "zipper/expression/binary/ArithmeticExpressions.hpp"
#include "zipper/expression/binary/ZeroAwareOperation.hpp"
#include "zipper/expression/unary/ScalarArithmetic.hpp"

namespace zipper {

UNARY_DECLARATION(VectorBase, LogicalNot, operator!)
UNARY_DECLARATION(VectorBase, BitNot, operator~)
UNARY_DECLARATION(VectorBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(VectorBase, Divides, operator/)

ZERO_AWARE_BINARY_DECLARATION(VectorBase, Plus, operator+)
ZERO_AWARE_BINARY_DECLARATION(VectorBase, Minus, operator-)

template <concepts::Vector Expr1, concepts::Vector Expr2>
auto operator==(Expr1 const &lhs, Expr2 const &rhs) -> bool {
    return (lhs.as_array() == rhs.as_array()).all();
}

template <concepts::Vector Expr1, concepts::Vector Expr2>
auto operator!=(Expr1 const &lhs, Expr2 const &rhs) -> bool {
    return (lhs.as_array() != rhs.as_array()).any();
}
SCALAR_BINARY_DECLARATION(VectorBase, Multiplies, operator*)

// Outer product: column vector times row vector (form) -> matrix.
template <concepts::Vector V, concepts::Form F>
auto operator*(V &&v, F &&f) {
    return std::forward<V>(v).outer(std::forward<F>(f));
}

// Free-function forms of dot product and norm.
// These enable ADL-based customization: external types can provide
// their own dot/norm in their namespace to satisfy the
// inner_product_space / normed_space concepts.

template <concepts::Vector V1, concepts::Vector V2>
auto dot(V1 const &a, V2 const &b) {
    return a.dot(b);
}

template <concepts::Vector V>
auto norm(V const &v) {
    return v.norm();
}

} // namespace zipper

#endif
