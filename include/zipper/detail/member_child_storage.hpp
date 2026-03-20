#pragma once

#include "ViewPropagating.hpp"

#include <type_traits>

namespace zipper::detail {

/// Computes the child expression storage type for member functions that use
/// deducing `this` (C++23 explicit object parameters).
///
/// Given the deduced `Self` type of the explicit object parameter and the
/// expression type stored in the wrapper, this trait selects:
///
///   - `ExprType`          when Self is an rvalue  → store by value (safe to return)
///   - `ExprType`          when ExprType is view-propagating → store by value
///                         (copying is cheap — just copies a reference wrapper)
///   - `const ExprType &`  when Self is a const lvalue  → store by const ref
///   - `ExprType &`        when Self is a mutable lvalue → store by mutable ref
///
/// The view-propagating case enables derived views (head, tail, row, etc.)
/// to be Returnable when the source was wrapped with ref().
template <typename Self, typename ExprType>
using member_child_storage_t = std::conditional_t<
    !std::is_lvalue_reference_v<Self> || ViewPropagating<ExprType>,
    ExprType,                                            // rvalue or view-propagating → by value
    std::conditional_t<
        std::is_const_v<std::remove_reference_t<Self>>,
        const ExprType &,                                // const lvalue → const ref
        ExprType &                                       // mutable lvalue → mutable ref
    >>;

} // namespace zipper::detail
