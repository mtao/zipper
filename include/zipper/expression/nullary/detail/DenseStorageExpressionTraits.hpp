#if !defined(ZIPPER_EXPRESSION_NULLARY_DETAIL_DENSESTORAGEEXPRESSIONTRAITS_HPP)
#define ZIPPER_EXPRESSION_NULLARY_DETAIL_DENSESTORAGEEXPRESSIONTRAITS_HPP
#include "zipper/detail//ExtentsTraits.hpp"
#include "zipper/expression/detail/ExpressionTraits.hpp"
#include "zipper/types.hpp"

namespace zipper::expression::nullary {

template <typename T> class DenseStorageExpressionBase;
}
namespace zipper::expression::detail {
template <typename T>
concept StorageExpressionDerived =
    std::derived_from<T, nullary::DenseStorageExpressionBase<T>>;
template <typename T> struct DenseStorageExpressionTraits;
//{
//    using value_type = ...;
//    using extents_type = ...;
//    using layout_policy = default_layout_policy;
//    using accessor_policy = default_accessor_policy<T>;
//};
// template <StorageExpressionDerived T>
// struct ExpressionTraits<T> : public StorageExpressionTraits<T> {
//    using Base = StorageExpressionTraits<T>;
//    using extents_type = Base::extents_type;
//    using extents_traits = Base::extents_traits;
//    using layout_policy = Base::layout_policy;
//    using mapping_type = typename layout_policy::template
//    mapping<extents_type>;
//};

template <typename Derived>
using mdspan_type =
    zipper::mdspan<typename ExpressionTraits<Derived>::value_type,
                   typename ExpressionTraits<Derived>::Extents,
                   typename ExpressionTraits<Derived>::layout_policy,
                   typename ExpressionTraits<Derived>::accessor_policy>;
} // namespace zipper::expression::detail
#endif
