#if !defined(ZIPPER_DETAIL_DECLARE_OPERATIONS_HPP)
#define ZIPPER_DETAIL_DECLARE_OPERATIONS_HPP

#include "zipper/concepts/ViewDerived.hpp"
#include "zipper/concepts/ZipperBaseDerived.hpp"
#include "zipper/views/binary/detail/operation_implementations.hpp"
#include "zipper/views/unary/detail/operation_implementations.hpp"

#define SCALAR_BINARY_DECLARATION(BASETYPE, NAME, OP)                \
    template <concepts::BASETYPE##Derived ViewType, typename Scalar> \
        requires(!concepts::BASETYPE##Derived<Scalar> &&             \
                 std::is_scalar_v<Scalar> &&                         \
                 !concepts::ZipperBaseDerived<Scalar>)               \
    auto OP(const ViewType& a, const Scalar& b) {                    \
        return views::unary::detail::operation_implementation<       \
            views::unary::Scalar##NAME##View, zipper::BASETYPE,      \
            const typename ViewType::view_type, Scalar>(a.view(), b);      \
    }                                                                \
    template <concepts::BASETYPE##Derived ViewType, typename Scalar> \
        requires(!concepts::BASETYPE##Derived<Scalar> &&             \
                 std::is_scalar_v<Scalar> &&                         \
                 !concepts::ZipperBaseDerived<Scalar>)               \
    auto OP(const Scalar& a, const ViewType& b) {                    \
        return views::unary::detail::operation_implementation<       \
            views::unary::Scalar##NAME##View, zipper::BASETYPE,      \
            const typename ViewType::view_type, Scalar>(a, b.view());      \
    }

#define UNARY_DECLARATION(BASETYPE, NAME, OP)                  \
    template <concepts::BASETYPE##Derived ViewType>            \
    auto OP(const ViewType& a) {                               \
        return views::unary::detail::operation_implementation< \
            views::unary::NAME##View, zipper::BASETYPE,        \
            const typename ViewType::view_type>(a.view());           \
    }

#define BINARY_DECLARATION(BASETYPE, NAME, OP)                            \
    template <concepts::BASETYPE##Derived ViewType,                       \
              concepts::BASETYPE##Derived ViewType2>                      \
    auto OP(const ViewType& a, const ViewType2& b) {                      \
        return views::binary::detail::operation_implementation<           \
            views::binary::NAME##View, zipper::BASETYPE,                  \
            const typename ViewType::view_type, const typename ViewType2::view_type>( \
            a.view(), b.view());                                          \
    }
#endif
