#if !defined(ZIPPER_ARRAYBASE_HXX)
#define ZIPPER_ARRAYBASE_HXX

#include "ArrayBase.hpp"
#include "expression/nullary/MDSpan.hpp"
#include "expression/unary/detail/operation_implementations.hpp"
#include "zipper/detail/declare_operations.hpp"
#include "zipper/expression/binary/ArithmeticExpressions.hpp"
#include "zipper/expression/unary/ScalarArithmetic.hpp"

namespace zipper {

// Deduction guide from std::mdspan
template <typename T, typename Extents, typename Layout, typename Accessor>
ArrayBase(zipper::mdspan<T, Extents, Layout, Accessor>) -> ArrayBase<
    expression::nullary::MDSpan<T, Extents, Layout, Accessor>>;

UNARY_DECLARATION(ArrayBase, LogicalNot, operator!)
UNARY_DECLARATION(ArrayBase, BitNot, operator~)
UNARY_DECLARATION(ArrayBase, Negate, operator-)

SCALAR_BINARY_DECLARATION(ArrayBase, Plus, operator+)
SCALAR_BINARY_DECLARATION(ArrayBase, Minus, operator-)
SCALAR_BINARY_DECLARATION(ArrayBase, Multiplies, operator*)
SCALAR_BINARY_DECLARATION(ArrayBase, Divides, operator/)
SCALAR_BINARY_DECLARATION(ArrayBase, Modulus, operator%)
SCALAR_BINARY_DECLARATION(ArrayBase, EqualsTo, operator==)
SCALAR_BINARY_DECLARATION(ArrayBase, NotEqualsTo, operator!=)
SCALAR_BINARY_DECLARATION(ArrayBase, Greater, operator>)
SCALAR_BINARY_DECLARATION(ArrayBase, Less, operator<)
SCALAR_BINARY_DECLARATION(ArrayBase, GreaterEqual, operator>=)
SCALAR_BINARY_DECLARATION(ArrayBase, LessEqual, operator<=)
// GCC notes that these operators don't allow short circuiting, but that's ok
// for our expressions
SCALAR_BINARY_DECLARATION(ArrayBase, BitAnd, operator&)
SCALAR_BINARY_DECLARATION(ArrayBase, BitOr, operator|)
SCALAR_BINARY_DECLARATION(ArrayBase, BitXor, operator^)

BINARY_DECLARATION(ArrayBase, Plus, operator+)
BINARY_DECLARATION(ArrayBase, Minus, operator-)
BINARY_DECLARATION(ArrayBase, Multiplies, operator*)
BINARY_DECLARATION(ArrayBase, Divides, operator/)
BINARY_DECLARATION(ArrayBase, Modulus, operator%)
BINARY_DECLARATION(ArrayBase, EqualsTo, operator==)
BINARY_DECLARATION(ArrayBase, NotEqualsTo, operator!=)
BINARY_DECLARATION(ArrayBase, Greater, operator>)
BINARY_DECLARATION(ArrayBase, Less, operator<)
BINARY_DECLARATION(ArrayBase, GreaterEqual, operator>=)
BINARY_DECLARATION(ArrayBase, LessEqual, operator<=)

// GCC notes that these operators don't allow short circuiting, but that's ok
// for our expressions
BINARY_DECLARATION(ArrayBase, BitAnd, operator&)
BINARY_DECLARATION(ArrayBase, BitOr, operator|)
BINARY_DECLARATION(ArrayBase, BitXor, operator^)

BINARY_DECLARATION(ArrayBase, Min, min)
BINARY_DECLARATION(ArrayBase, Max, max)

} // namespace zipper

#endif
