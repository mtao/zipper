#if !defined(ZIPPER_VIEWS_UNARY_IDEMPOTENTVIEW_HPP)
#define ZIPPER_VIEWS_UNARY_IDEMPOTENTVIEW_HPP


#include "OperationView.hpp"
#include <functional>

namespace zipper::views::unary {


template <concepts::ViewDerived B>
using IdentityView = OperationView<
    B, std::identity>;

}  // namespace zipper::views::unary
#endif
