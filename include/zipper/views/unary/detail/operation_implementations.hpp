#if !defined(ZIPPER_VIEWS_UNARY_DETAIL_SCALAR_BINARY_OPERATION_DECLARATION_HPP)

#define ZIPPER_VIEWS_UNARY_DETAIL_SCALAR_BINARY_OPERATION_DECLARATION_HPP

#include "zipper/concepts/ViewDerived.hpp"

namespace zipper::views::unary::detail {
template <template <typename,zipper::concepts::QualifiedViewDerived, bool> class Op,
          template <typename> class BaseType,zipper::concepts::QualifiedViewDerived View,
          typename Scalar>
auto operation_implementation(View& lhs, const Scalar& rhs) {
    using OpType = Op<Scalar, View, true>;
    return BaseType<OpType>(OpType(lhs, rhs, {}));
}
template <template <typename,zipper::concepts::QualifiedViewDerived, bool> class Op,
          template <typename> class BaseType,zipper::concepts::QualifiedViewDerived View,
          typename Scalar>
auto operation_implementation(const Scalar& lhs, View& rhs) {
    using OpType = Op<std::decay_t<Scalar>, View, false>;
    return BaseType<OpType>(OpType(lhs, rhs, {}));
}

template <template <zipper::concepts::QualifiedViewDerived> class Op,
          template <typename> class BaseType,zipper::concepts::QualifiedViewDerived B>
auto operation_implementation(B& view) {
    using OpType = Op<B>;
    return BaseType<OpType>(OpType(view));
}

}  // namespace zipper::views::unary::detail
#endif
