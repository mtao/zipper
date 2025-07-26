#if !defined(ZIPPER_VIEWS_UNARY_DETAIL_SCALAR_BINARY_OPERATION_DECLARATION_HPP)

#define ZIPPER_VIEWS_UNARY_DETAIL_SCALAR_BINARY_OPERATION_DECLARATION_HPP

#include "zipper/concepts/ViewDerived.hpp"

namespace zipper::views::unary::detail {
template <template <typename,zipper::concepts::ViewDerived, bool> class Op,
          template <typename> class BaseType,zipper::concepts::ViewDerived View,
          typename Scalar>
auto operation_implementation(const View& lhs, const Scalar& rhs) {
    using OpType = Op<Scalar, View, true>;
    return BaseType<OpType>(OpType(lhs, rhs, {}));
}
template <template <typename,zipper::concepts::ViewDerived, bool> class Op,
          template <typename> class BaseType,zipper::concepts::ViewDerived View,
          typename Scalar>
auto operation_implementation(const Scalar& lhs, const View& rhs) {
    using OpType = Op<std::decay_t<Scalar>, std::decay_t<View>, false>;
    return BaseType<OpType>(OpType(lhs, rhs, {}));
}

template <template <zipper::concepts::ViewDerived> class Op,
          template <typename> class BaseType,zipper::concepts::ViewDerived B>
auto operation_implementation(const B& view) {
    using OpType = Op<B>;
    return BaseType<OpType>(OpType(view));
}

}  // namespace zipper::views::unary::detail
#endif
