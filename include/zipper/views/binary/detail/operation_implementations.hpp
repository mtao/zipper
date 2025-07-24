#if !defined(ZIPPER_VIEWS_BINARY_DETAIL_SCALAR_BINARY_OPERATION_DECLARATION_HPP)

#define ZIPPER_VIEWS_BINARY_DETAIL_SCALAR_BINARY_OPERATION_DECLARATION_HPP

#include "zipper/concepts/ViewDerived.hpp"

namespace zipper::views::binary::detail {
template <template <zipper::concepts::ViewDerived,
                    zipper::concepts::ViewDerived> class Op,
          template <typename> class BaseType,
          zipper::concepts::ViewDerived ViewA,
          zipper::concepts::ViewDerived ViewB>
auto operation_implementation(const ViewA& lhs, const ViewB& rhs) {
    using OpType = Op<ViewA, ViewB>;
    return BaseType<OpType>(OpType(lhs, rhs, {}));
}

}  // namespace zipper::views::binary::detail
#endif
