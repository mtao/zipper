#if !defined(ZIPPER_EXPRESSION_UNARY_IDENTITY_HPP)
#define ZIPPER_EXPRESSION_UNARY_IDENTITY_HPP

#include "CoefficientWiseOperation.hpp"
#include <functional>

namespace zipper::expression::unary {

template <zipper::concepts::Expression Child>
using Identity = CoefficientWiseOperation<Child, std::identity>;

} // namespace zipper::expression::unary
#endif
