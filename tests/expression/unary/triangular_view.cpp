// Tests for TriangularView expression:
// - Lower / Upper read access
// - UnitLower / UnitUpper diagonal handling
// - StrictlyLower / StrictlyUpper (ZeroDiag)
// - Extents correctness (rank-2 preserved, same as child)
// - make_owned() materialization
// - Dynamic-extent matrices
// - Factory function triangular_view<Mode>()

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/unary/TriangularView.hpp>

#include "../../catch_include.hpp"

using namespace zipper;
using namespace zipper::expression;

// ─── Helper: 3x3 matrix with distinct entries ───────────────────────────────
// M = [1  2  3]
//     [4  5  6]
//     [7  8  9]
static auto make_3x3() -> Matrix<double, 3, 3> {
    return Matrix<double, 3, 3>{
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
}

// ─── Lower ──────────────────────────────────────────────────────────────────

TEST_CASE("triangular_view_lower_read", "[triangular_view][lower]") {
    auto M = make_3x3();
    auto L = triangular_view<TriangularMode::Lower>(M);

    // Extents should match the child
    REQUIRE(L.extents().rank() == 2);
    REQUIRE(L.extent(0) == 3);
    REQUIRE(L.extent(1) == 3);

    // Diagonal: pass-through
    CHECK(L.coeff(0ul, 0ul) == 1.0);
    CHECK(L.coeff(1ul, 1ul) == 5.0);
    CHECK(L.coeff(2ul, 2ul) == 9.0);

    // Lower triangle (row > col): pass-through
    CHECK(L.coeff(1ul, 0ul) == 4.0);
    CHECK(L.coeff(2ul, 0ul) == 7.0);
    CHECK(L.coeff(2ul, 1ul) == 8.0);

    // Upper triangle (row < col): zero
    CHECK(L.coeff(0ul, 1ul) == 0.0);
    CHECK(L.coeff(0ul, 2ul) == 0.0);
    CHECK(L.coeff(1ul, 2ul) == 0.0);
}

// ─── Upper ──────────────────────────────────────────────────────────────────

TEST_CASE("triangular_view_upper_read", "[triangular_view][upper]") {
    auto M = make_3x3();
    auto U = triangular_view<TriangularMode::Upper>(M);

    // Diagonal: pass-through
    CHECK(U.coeff(0ul, 0ul) == 1.0);
    CHECK(U.coeff(1ul, 1ul) == 5.0);
    CHECK(U.coeff(2ul, 2ul) == 9.0);

    // Upper triangle (row < col): pass-through
    CHECK(U.coeff(0ul, 1ul) == 2.0);
    CHECK(U.coeff(0ul, 2ul) == 3.0);
    CHECK(U.coeff(1ul, 2ul) == 6.0);

    // Lower triangle (row > col): zero
    CHECK(U.coeff(1ul, 0ul) == 0.0);
    CHECK(U.coeff(2ul, 0ul) == 0.0);
    CHECK(U.coeff(2ul, 1ul) == 0.0);
}

// ─── UnitLower ──────────────────────────────────────────────────────────────

TEST_CASE("triangular_view_unit_lower", "[triangular_view][unit_lower]") {
    auto M = make_3x3();
    auto UL = triangular_view<TriangularMode::UnitLower>(M);

    // Diagonal: forced to 1
    CHECK(UL.coeff(0ul, 0ul) == 1.0);
    CHECK(UL.coeff(1ul, 1ul) == 1.0);
    CHECK(UL.coeff(2ul, 2ul) == 1.0);

    // Lower triangle: pass-through
    CHECK(UL.coeff(1ul, 0ul) == 4.0);
    CHECK(UL.coeff(2ul, 0ul) == 7.0);
    CHECK(UL.coeff(2ul, 1ul) == 8.0);

    // Upper triangle: zero
    CHECK(UL.coeff(0ul, 1ul) == 0.0);
    CHECK(UL.coeff(0ul, 2ul) == 0.0);
    CHECK(UL.coeff(1ul, 2ul) == 0.0);
}

// ─── UnitUpper ──────────────────────────────────────────────────────────────

TEST_CASE("triangular_view_unit_upper", "[triangular_view][unit_upper]") {
    auto M = make_3x3();
    auto UU = triangular_view<TriangularMode::UnitUpper>(M);

    // Diagonal: forced to 1
    CHECK(UU.coeff(0ul, 0ul) == 1.0);
    CHECK(UU.coeff(1ul, 1ul) == 1.0);
    CHECK(UU.coeff(2ul, 2ul) == 1.0);

    // Upper triangle: pass-through
    CHECK(UU.coeff(0ul, 1ul) == 2.0);
    CHECK(UU.coeff(0ul, 2ul) == 3.0);
    CHECK(UU.coeff(1ul, 2ul) == 6.0);

    // Lower triangle: zero
    CHECK(UU.coeff(1ul, 0ul) == 0.0);
    CHECK(UU.coeff(2ul, 0ul) == 0.0);
    CHECK(UU.coeff(2ul, 1ul) == 0.0);
}

// ─── StrictlyLower ──────────────────────────────────────────────────────────

TEST_CASE("triangular_view_strictly_lower", "[triangular_view][strictly_lower]") {
    auto M = make_3x3();
    auto SL = triangular_view<TriangularMode::StrictlyLower>(M);

    // Diagonal: forced to 0
    CHECK(SL.coeff(0ul, 0ul) == 0.0);
    CHECK(SL.coeff(1ul, 1ul) == 0.0);
    CHECK(SL.coeff(2ul, 2ul) == 0.0);

    // Lower triangle: pass-through
    CHECK(SL.coeff(1ul, 0ul) == 4.0);
    CHECK(SL.coeff(2ul, 0ul) == 7.0);
    CHECK(SL.coeff(2ul, 1ul) == 8.0);

    // Upper triangle: zero
    CHECK(SL.coeff(0ul, 1ul) == 0.0);
    CHECK(SL.coeff(0ul, 2ul) == 0.0);
    CHECK(SL.coeff(1ul, 2ul) == 0.0);
}

// ─── StrictlyUpper ──────────────────────────────────────────────────────────

TEST_CASE("triangular_view_strictly_upper", "[triangular_view][strictly_upper]") {
    auto M = make_3x3();
    auto SU = triangular_view<TriangularMode::StrictlyUpper>(M);

    // Diagonal: forced to 0
    CHECK(SU.coeff(0ul, 0ul) == 0.0);
    CHECK(SU.coeff(1ul, 1ul) == 0.0);
    CHECK(SU.coeff(2ul, 2ul) == 0.0);

    // Upper triangle: pass-through
    CHECK(SU.coeff(0ul, 1ul) == 2.0);
    CHECK(SU.coeff(0ul, 2ul) == 3.0);
    CHECK(SU.coeff(1ul, 2ul) == 6.0);

    // Lower triangle: zero
    CHECK(SU.coeff(1ul, 0ul) == 0.0);
    CHECK(SU.coeff(2ul, 0ul) == 0.0);
    CHECK(SU.coeff(2ul, 1ul) == 0.0);
}

// ─── View reflects mutations ────────────────────────────────────────────────

TEST_CASE("triangular_view_reflects_mutations", "[triangular_view][view]") {
    auto M = make_3x3();
    auto L = triangular_view<TriangularMode::Lower>(M);

    // Initial check
    CHECK(L.coeff(2ul, 0ul) == 7.0);

    // Mutate M and verify the view sees the change
    M(2, 0) = 42.0;
    CHECK(L.coeff(2ul, 0ul) == 42.0);

    // Mutation to an upper-triangle entry doesn't change the view
    M(0, 2) = 999.0;
    CHECK(L.coeff(0ul, 2ul) == 0.0);
}

// ─── make_owned materializes ────────────────────────────────────────────────

TEST_CASE("triangular_view_make_owned", "[triangular_view][owned]") {
    auto M = make_3x3();
    auto L = triangular_view<TriangularMode::Lower>(M);
    auto owned = L.make_owned();

    // Verify the owned copy has correct values
    CHECK(owned.coeff(0ul, 0ul) == 1.0);
    CHECK(owned.coeff(1ul, 0ul) == 4.0);
    CHECK(owned.coeff(0ul, 1ul) == 0.0);

    // Mutate M — owned copy should be independent
    M(1, 0) = 999.0;
    CHECK(owned.coeff(1ul, 0ul) == 4.0);  // still the old value
}

// ─── Non-square matrices ────────────────────────────────────────────────────

TEST_CASE("triangular_view_non_square", "[triangular_view][non_square]") {
    // 2x4 matrix
    // M = [1  2  3  4]
    //     [5  6  7  8]
    Matrix<double, 2, 4> M;
    M(0, 0) = 1.0; M(0, 1) = 2.0; M(0, 2) = 3.0; M(0, 3) = 4.0;
    M(1, 0) = 5.0; M(1, 1) = 6.0; M(1, 2) = 7.0; M(1, 3) = 8.0;

    auto L = triangular_view<TriangularMode::Lower>(M);

    // Row 0: diagonal at (0,0), rest is upper
    CHECK(L.coeff(0ul, 0ul) == 1.0);  // diagonal
    CHECK(L.coeff(0ul, 1ul) == 0.0);  // upper → 0
    CHECK(L.coeff(0ul, 2ul) == 0.0);  // upper → 0
    CHECK(L.coeff(0ul, 3ul) == 0.0);  // upper → 0

    // Row 1: (1,0) is lower, (1,1) is diagonal, (1,2) and (1,3) are upper
    CHECK(L.coeff(1ul, 0ul) == 5.0);  // lower
    CHECK(L.coeff(1ul, 1ul) == 6.0);  // diagonal
    CHECK(L.coeff(1ul, 2ul) == 0.0);  // upper → 0
    CHECK(L.coeff(1ul, 3ul) == 0.0);  // upper → 0

    auto U = triangular_view<TriangularMode::Upper>(M);

    // Row 0: diagonal + upper
    CHECK(U.coeff(0ul, 0ul) == 1.0);
    CHECK(U.coeff(0ul, 1ul) == 2.0);
    CHECK(U.coeff(0ul, 2ul) == 3.0);
    CHECK(U.coeff(0ul, 3ul) == 4.0);

    // Row 1: (1,0) is lower → 0
    CHECK(U.coeff(1ul, 0ul) == 0.0);
    CHECK(U.coeff(1ul, 1ul) == 6.0);
    CHECK(U.coeff(1ul, 2ul) == 7.0);
    CHECK(U.coeff(1ul, 3ul) == 8.0);
}

// ─── Dynamic-extent matrix ──────────────────────────────────────────────────

TEST_CASE("triangular_view_dynamic", "[triangular_view][dynamic]") {
    Matrix<double, std::dynamic_extent, std::dynamic_extent> M(3, 3);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            M(i, j) = static_cast<double>(i * 10 + j);
        }
    }
    // M = [ 0  1  2]
    //     [10 11 12]
    //     [20 21 22]

    auto L = triangular_view<TriangularMode::Lower>(M);

    REQUIRE(L.extent(0) == 3);
    REQUIRE(L.extent(1) == 3);

    // Diagonal
    CHECK(L.coeff(0ul, 0ul) == 0.0);
    CHECK(L.coeff(1ul, 1ul) == 11.0);
    CHECK(L.coeff(2ul, 2ul) == 22.0);

    // Lower
    CHECK(L.coeff(1ul, 0ul) == 10.0);
    CHECK(L.coeff(2ul, 0ul) == 20.0);
    CHECK(L.coeff(2ul, 1ul) == 21.0);

    // Upper → zero
    CHECK(L.coeff(0ul, 1ul) == 0.0);
    CHECK(L.coeff(0ul, 2ul) == 0.0);
    CHECK(L.coeff(1ul, 2ul) == 0.0);
}

// ─── 1x1 matrix (edge case) ────────────────────────────────────────────────

TEST_CASE("triangular_view_1x1", "[triangular_view][edge]") {
    Matrix<double, 1, 1> M;
    M(0, 0) = 42.0;

    auto L = triangular_view<TriangularMode::Lower>(M);
    CHECK(L.coeff(0ul, 0ul) == 42.0);

    auto UL = triangular_view<TriangularMode::UnitLower>(M);
    CHECK(UL.coeff(0ul, 0ul) == 1.0);

    auto SL = triangular_view<TriangularMode::StrictlyLower>(M);
    CHECK(SL.coeff(0ul, 0ul) == 0.0);

    auto U = triangular_view<TriangularMode::Upper>(M);
    CHECK(U.coeff(0ul, 0ul) == 42.0);

    auto UU = triangular_view<TriangularMode::UnitUpper>(M);
    CHECK(UU.coeff(0ul, 0ul) == 1.0);

    auto SU = triangular_view<TriangularMode::StrictlyUpper>(M);
    CHECK(SU.coeff(0ul, 0ul) == 0.0);
}

// ─── 4x4 Lower + Upper = original (for non-unit/non-strict modes) ──────────

TEST_CASE("triangular_view_lower_plus_upper_reconstruction",
          "[triangular_view][reconstruction]") {
    // For plain Lower and Upper (pass-through diagonal), the sum
    // L + U should equal M + diag(M) because the diagonal is counted
    // in both.  Instead, verify L + StrictlyUpper = M.
    auto M = make_3x3();
    auto L = triangular_view<TriangularMode::Lower>(M);
    auto SU = triangular_view<TriangularMode::StrictlyUpper>(M);

    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double sum = L.coeff(i, j) + SU.coeff(i, j);
            CHECK(sum == M(i, j));
        }
    }
}

// ─── Const matrix input ─────────────────────────────────────────────────────

TEST_CASE("triangular_view_const_matrix", "[triangular_view][const]") {
    const auto M = make_3x3();
    auto L = triangular_view<TriangularMode::Lower>(M);

    CHECK(L.coeff(0ul, 0ul) == 1.0);
    CHECK(L.coeff(2ul, 1ul) == 8.0);
    CHECK(L.coeff(0ul, 2ul) == 0.0);
}
