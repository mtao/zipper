#if !defined(UVL_VIEWS_BINARY_DETAIL_SCALAR_BINARY_OPERATION_DECLARATION_HPP)

#define UVL_VIEWS_BINARY_DETAIL_SCALAR_BINARY_OPERATION_DECLARATION_HPP

#include "uvl/concepts/ViewDerived.hpp"

namespace uvl::views::binary::detail {
template <template <concepts::ViewDerived, concepts::ViewDerived> class Op,
          template <typename> class BaseType, concepts::ViewDerived ViewA,
          concepts::ViewDerived ViewB>
auto operation_implementation(const ViewA& lhs, const ViewB& rhs) {
    using OpType = Op<ViewA, ViewB>;
    return BaseType<OpType>(OpType(lhs, rhs, {}));
}

}  // namespace uvl::views::binary::detail
#endif
