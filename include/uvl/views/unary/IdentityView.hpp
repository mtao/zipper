#if !defined(UVL_VIEWS_UNARY_IDEMPOTENTVIEW_HPP)
#define UVL_VIEWS_UNARY_IDEMPOTENTVIEW_HPP


#include "OperationView.hpp"
#include <functional>

namespace uvl::views::unary {


template <concepts::ViewDerived B>
using IdentityView = OperationView<
    B, std::identity>;

}  // namespace uvl::views::unary
#endif
