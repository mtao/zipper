#if !defined(UVL_VIEWS_UNARY_DETAIL_SCALAR_BINARY_OPERATION_DECLARATION_HPP)

#define UVL_VIEWS_UNARY_DETAIL_SCALAR_BINARY_OPERATION_DECLARATION_HPP

#include "uvl/concepts/ViewDerived.hpp"

namespace uvl::views::unary::detail {
template <template <typename, concepts::ViewDerived, bool> class Op,
          template <typename> class BaseType, concepts::ViewDerived View,
          typename Scalar>
auto operation_implementation(const View& lhs, const Scalar& rhs) {
    using OpType = Op<Scalar, View, true>;
    return BaseType<OpType>(OpType(lhs, rhs, {}));
}
template <template <typename, concepts::ViewDerived, bool> class Op,
          template <typename> class BaseType, concepts::ViewDerived View,
          typename Scalar>
auto operation_implementation(const Scalar& lhs, const View& rhs) {
    using OpType = Op<std::decay_t<Scalar>, std::decay_t<View>, false>;
    return BaseType<OpType>(OpType(lhs, rhs, {}));
}

template <template <concepts::ViewDerived> class Op,
          template <typename> class BaseType, concepts::ViewDerived B>
auto operation_implementation(const B& view) {
    using OpType = Op<B>;
    return BaseType<OpType>(OpType(view));
}

}  // namespace uvl::views::unary::detail
#endif
