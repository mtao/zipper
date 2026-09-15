#if !defined(ZIPPER_DETAIL_MEMBER_CHILD_STORAGE_HPP)
#define ZIPPER_DETAIL_MEMBER_CHILD_STORAGE_HPP

#include "ViewPropagating.hpp"

#include <type_traits>

namespace zipper::detail {

/// Computes the child expression storage type for member functions that use
/// deducing `this` (C++23 explicit object parameters).
///
/// Given the deduced `Self` type of the explicit object parameter and the
/// expression type stored in the wrapper, this trait selects:
///
/// ExprType must be the raw (possibly cvref-qualified) stored expression type.
/// An rvalue wrapper owns its child only if it owns its expression; moving a
/// reference-backed wrapper must not move or copy the referent. Const on either
/// the wrapper or the referent is preserved, including for by-value children.
///
/// The view-propagating case enables derived views (head, tail, row, etc.)
/// to be Returnable when the source was wrapped with ref().
template <typename Self, typename ExprType>
struct member_child_storage {
    using qualified_expression = std::conditional_t<
        std::is_const_v<std::remove_reference_t<Self>>,
        const std::remove_reference_t<ExprType>,
        std::remove_reference_t<ExprType>>;
    using type = std::conditional_t<
        ViewPropagating<ExprType> ||
            (!std::is_lvalue_reference_v<Self> &&
             !std::is_reference_v<ExprType>),
        qualified_expression, qualified_expression &>;
};

template <typename Self, typename ExprType>
using member_child_storage_t = typename member_child_storage<Self, ExprType>::type;

} // namespace zipper::detail
#endif
