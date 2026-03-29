#if !defined(ZIPPER_EXPRESSION_UNARY_DIAGONALMATRIX_HPP)
#define ZIPPER_EXPRESSION_UNARY_DIAGONALMATRIX_HPP

/// @file DiagonalMatrix.hpp
/// @brief Backward-compatibility forwarding header.
/// @ingroup expressions_unary
///
/// This header forwards to `DiagonalEmbed.hpp`.  New code should include
/// `<zipper/expression/unary/DiagonalEmbed.hpp>` directly.

#include "DiagonalEmbed.hpp"

namespace zipper::expression::unary {

/// @brief Backward-compatibility alias.
template <zipper::concepts::QualifiedExpression ExpressionType>
using DiagonalMatrix = DiagonalEmbed<ExpressionType>;

} // namespace zipper::expression::unary

#endif
