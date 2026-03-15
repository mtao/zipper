/// @file TriangularMode.hpp
/// @brief Bitmask enum for triangular view modes and related utilities.
///
/// Extracted from TriangularView.hpp so that solver detail headers can
/// reference the mode flags without pulling in the full TriangularView
/// class definition (avoiding circular includes).
///
/// @see zipper::expression::unary::TriangularView

#pragma once

namespace zipper::expression {

/// @brief Bitmask flags that compose to form a TriangularMode.
///
/// The four fundamental bits are:
///   - Lower   (0x1): include the lower triangle (row >= col)
///   - Upper   (0x2): include the upper triangle (row <= col)
///   - UnitDiag(0x4): force diagonal elements to 1
///   - ZeroDiag(0x8): force diagonal elements to 0
///
/// Composite modes combine a triangle bit with a diagonal bit:
///   - UnitLower     = Lower  | UnitDiag   (lower triangle with unit diagonal)
///   - UnitUpper     = Upper  | UnitDiag   (upper triangle with unit diagonal)
///   - StrictlyLower = Lower  | ZeroDiag   (lower triangle excluding diagonal)
///   - StrictlyUpper = Upper  | ZeroDiag   (upper triangle excluding diagonal)
///
/// Using Lower or Upper alone passes through the child's diagonal values.
enum class TriangularMode : unsigned {
    Lower   = 0x1,
    Upper   = 0x2,
    UnitDiag = 0x4,
    ZeroDiag = 0x8,

    UnitLower     = Lower | UnitDiag,
    UnitUpper     = Upper | UnitDiag,
    StrictlyLower = Lower | ZeroDiag,
    StrictlyUpper = Upper | ZeroDiag,
};

/// @brief Bitwise OR for TriangularMode flags.
constexpr auto operator|(TriangularMode a, TriangularMode b) -> TriangularMode {
    return static_cast<TriangularMode>(
        static_cast<unsigned>(a) | static_cast<unsigned>(b));
}

/// @brief Bitwise AND for TriangularMode flags.
constexpr auto operator&(TriangularMode a, TriangularMode b) -> TriangularMode {
    return static_cast<TriangularMode>(
        static_cast<unsigned>(a) & static_cast<unsigned>(b));
}

/// @brief Test whether a specific bit is set in a TriangularMode.
constexpr auto has_flag(TriangularMode mode, TriangularMode flag) -> bool {
    return (mode & flag) == flag;
}

/// @brief Checks that a TriangularMode value is well-formed:
///   - Exactly one of Lower or Upper is set.
///   - UnitDiag and ZeroDiag are not both set.
template <TriangularMode Mode>
consteval auto is_valid_triangular_mode() -> bool {
    return (has_flag(Mode, TriangularMode::Lower) !=
            has_flag(Mode, TriangularMode::Upper)) &&
           !(has_flag(Mode, TriangularMode::UnitDiag) &&
             has_flag(Mode, TriangularMode::ZeroDiag));
}

} // namespace zipper::expression
