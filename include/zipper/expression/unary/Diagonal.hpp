#if !defined(ZIPPER_expression_UNARY_DIAGONALVIEW_HPP)
#define ZIPPER_expression_UNARY_DIAGONALVIEW_HPP

/// @file Diagonal.hpp
/// @brief Backward-compatibility forwarding header.
/// @ingroup expressions_unary
///
/// This header forwards to `DiagonalExtract.hpp`.  New code should include
/// `<zipper/expression/unary/DiagonalExtract.hpp>` directly.

#include "DiagonalExtract.hpp"

namespace zipper::expression::unary {

/// @brief Backward-compatibility alias.
template <zipper::concepts::QualifiedExpression ExpressionType>
using Diagonal = DiagonalExtract<ExpressionType>;

} // namespace zipper::expression::unary

#endif
