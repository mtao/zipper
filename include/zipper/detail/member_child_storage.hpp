#if !defined(ZIPPER_DETAIL_MEMBER_CHILD_STORAGE_HPP)
#define ZIPPER_DETAIL_MEMBER_CHILD_STORAGE_HPP

#include <type_traits>

namespace zipper::detail {

/// Computes the child expression storage type for member functions that use
/// deducing `this` (C++23 explicit object parameters).
///
/// Given the deduced `Self` type of the explicit object parameter and the
/// expression type stored in the wrapper, this trait selects:
///
///   - `ExprType`          when Self is an rvalue  → store by value (safe to return)
///   - `const ExprType &`  when Self is a const lvalue  → store by const ref
///   - `ExprType &`        when Self is a mutable lvalue → store by mutable ref
///
/// This ensures that expression views built from temporaries own their
/// children (preventing dangling references), while views built from
/// named objects store lightweight references as before.
template <typename Self, typename ExprType>
using member_child_storage_t = std::conditional_t<
    !std::is_lvalue_reference_v<Self>,
    ExprType,                                            // rvalue → by value
    std::conditional_t<
        std::is_const_v<std::remove_reference_t<Self>>,
        const ExprType &,                                // const lvalue → const ref
        ExprType &                                       // mutable lvalue → mutable ref
    >>;

} // namespace zipper::detail
#endif
