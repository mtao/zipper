// Tests for triangular_solve (forward/back substitution):
// - Lower triangular solve (3x3)
// - Upper triangular solve (3x3)
// - UnitLower and UnitUpper solves
// - StrictlyLower and StrictlyUpper (should fail — singular)
// - Zero pivot detection
// - In-place variant
// - Dynamic-extent vectors

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/unary/TriangularView.hpp>
#include <zipper/utils/solver/triangular_solve.hpp>

#include "../catch_include.hpp"

using namespace zipper;
using namespace zipper::expression;
using namespace zipper::utils::solver;

// ═══════════════════════════════════════════════════════════════════════════════
// Lower triangular solve (forward substitution)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("triangular_solve lower 3x3", "[triangular_solve][lower]") {
    // L = [2  0  0]    b = [2]     x = [1]
    //     [1  3  0]        [7]         [2]
    //     [4  2  1]        [11]        [3]
    Matrix<double, 3, 3> M{{2.0, 0.0, 0.0},
                           {1.0, 3.0, 0.0},
                           {4.0, 2.0, 1.0}};
    Vector<double, 3> b{2.0, 7.0, 11.0};

    auto L = triangular_view<TriangularMode::Lower>(M);
    auto result = triangular_solve(L, b);

    REQUIRE(result.has_value());
    CHECK((*result)(0ul) == Catch::Approx(1.0).epsilon(1e-12));
    CHECK((*result)(1ul) == Catch::Approx(2.0).epsilon(1e-12));
    CHECK((*result)(2ul) == Catch::Approx(3.0).epsilon(1e-12));
}

TEST_CASE("triangular_solve lower 2x2", "[triangular_solve][lower]") {
    // L = [4  0]    b = [8]     x = [2]
    //     [3  5]        [16]        [2]
    Matrix<double, 2, 2> M{{4.0, 0.0}, {3.0, 5.0}};
    Vector<double, 2> b{8.0, 16.0};

    auto L = triangular_view<TriangularMode::Lower>(M);
    auto result = triangular_solve(L, b);

    REQUIRE(result.has_value());
    CHECK((*result)(0ul) == Catch::Approx(2.0).epsilon(1e-12));
    CHECK((*result)(1ul) == Catch::Approx(2.0).epsilon(1e-12));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Upper triangular solve (back substitution)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("triangular_solve upper 3x3", "[triangular_solve][upper]") {
    // U = [3  1  2]    b = [10]    x = [1]
    //     [0  4  1]        [9]         [2]
    //     [0  0  2]        [2]         [1]
    //
    // Check: 3*1 + 1*2 + 2*1 = 7... Let me redo:
    // x = [1, 2, 1]
    // row 2: 2*1 = 2 => b[2] = 2. OK.
    // row 1: 4*2 + 1*1 = 9 => b[1] = 9. OK.
    // row 0: 3*1 + 1*2 + 2*1 = 7 => b[0] = 7.
    Matrix<double, 3, 3> M{{3.0, 1.0, 2.0},
                           {0.0, 4.0, 1.0},
                           {0.0, 0.0, 2.0}};
    Vector<double, 3> b{7.0, 9.0, 2.0};

    auto U = triangular_view<TriangularMode::Upper>(M);
    auto result = triangular_solve(U, b);

    REQUIRE(result.has_value());
    CHECK((*result)(0ul) == Catch::Approx(1.0).epsilon(1e-12));
    CHECK((*result)(1ul) == Catch::Approx(2.0).epsilon(1e-12));
    CHECK((*result)(2ul) == Catch::Approx(1.0).epsilon(1e-12));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit diagonal variants
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("triangular_solve unit_lower 3x3",
          "[triangular_solve][unit_lower]") {
    // UnitLower: diagonal forced to 1.
    // L = [1  0  0]    b = [1]    x = [1]
    //     [2  1  0]        [5]        [3]
    //     [3  1  1]        [10]       [4]
    //
    // x[0] = b[0]/1 = 1
    // x[1] = (b[1] - 2*1)/1 = 3
    // x[2] = (b[2] - 3*1 - 1*3)/1 = 4
    Matrix<double, 3, 3> M{{99.0, 0.0, 0.0},   // diagonal will be overridden to 1
                           {2.0, 99.0, 0.0},
                           {3.0, 1.0, 99.0}};
    Vector<double, 3> b{1.0, 5.0, 10.0};

    auto L = triangular_view<TriangularMode::UnitLower>(M);
    auto result = triangular_solve(L, b);

    REQUIRE(result.has_value());
    CHECK((*result)(0ul) == Catch::Approx(1.0).epsilon(1e-12));
    CHECK((*result)(1ul) == Catch::Approx(3.0).epsilon(1e-12));
    CHECK((*result)(2ul) == Catch::Approx(4.0).epsilon(1e-12));
}

TEST_CASE("triangular_solve unit_upper 3x3",
          "[triangular_solve][unit_upper]") {
    // UnitUpper: diagonal forced to 1.
    // U = [1  2  1]    x = [1, 2, 3]
    //     [0  1  3]    b[2] = 1*3 = 3
    //     [0  0  1]    b[1] = 1*2 + 3*3 = 11
    //                  b[0] = 1*1 + 2*2 + 1*3 = 8
    Matrix<double, 3, 3> M{{99.0, 2.0, 1.0},
                           {0.0, 99.0, 3.0},
                           {0.0, 0.0, 99.0}};
    Vector<double, 3> b{8.0, 11.0, 3.0};

    auto U = triangular_view<TriangularMode::UnitUpper>(M);
    auto result = triangular_solve(U, b);

    REQUIRE(result.has_value());
    CHECK((*result)(0ul) == Catch::Approx(1.0).epsilon(1e-12));
    CHECK((*result)(1ul) == Catch::Approx(2.0).epsilon(1e-12));
    CHECK((*result)(2ul) == Catch::Approx(3.0).epsilon(1e-12));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Strictly triangular (should fail — zero diagonal)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("triangular_solve strictly_lower fails",
          "[triangular_solve][strictly_lower]") {
    Matrix<double, 3, 3> M{{1.0, 0.0, 0.0},
                           {2.0, 3.0, 0.0},
                           {4.0, 5.0, 6.0}};
    Vector<double, 3> b{1.0, 2.0, 3.0};

    auto SL = triangular_view<TriangularMode::StrictlyLower>(M);
    auto result = triangular_solve(SL, b);

    REQUIRE(!result.has_value());
    CHECK(result.error().kind == SolverError::Kind::breakdown);
}

TEST_CASE("triangular_solve strictly_upper fails",
          "[triangular_solve][strictly_upper]") {
    Matrix<double, 3, 3> M{{1.0, 2.0, 3.0},
                           {0.0, 4.0, 5.0},
                           {0.0, 0.0, 6.0}};
    Vector<double, 3> b{1.0, 2.0, 3.0};

    auto SU = triangular_view<TriangularMode::StrictlyUpper>(M);
    auto result = triangular_solve(SU, b);

    REQUIRE(!result.has_value());
    CHECK(result.error().kind == SolverError::Kind::breakdown);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Zero pivot detection
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("triangular_solve zero pivot", "[triangular_solve][breakdown]") {
    // Lower with zero on diagonal at position (1,1).
    Matrix<double, 3, 3> M{{2.0, 0.0, 0.0},
                           {1.0, 0.0, 0.0},  // zero pivot!
                           {4.0, 2.0, 1.0}};
    Vector<double, 3> b{2.0, 1.0, 7.0};

    auto L = triangular_view<TriangularMode::Lower>(M);
    auto result = triangular_solve(L, b);

    REQUIRE(!result.has_value());
    CHECK(result.error().kind == SolverError::Kind::breakdown);
}

// ═══════════════════════════════════════════════════════════════════════════════
// In-place variant
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("triangular_solve_inplace lower",
          "[triangular_solve][inplace]") {
    Matrix<double, 3, 3> M{{2.0, 0.0, 0.0},
                           {1.0, 3.0, 0.0},
                           {4.0, 2.0, 1.0}};
    Vector<double, 3> x{2.0, 7.0, 11.0};  // starts as b

    auto L = triangular_view<TriangularMode::Lower>(M);
    auto status = triangular_solve_inplace(L, x);

    REQUIRE(status.has_value());
    CHECK(x(0ul) == Catch::Approx(1.0).epsilon(1e-12));
    CHECK(x(1ul) == Catch::Approx(2.0).epsilon(1e-12));
    CHECK(x(2ul) == Catch::Approx(3.0).epsilon(1e-12));
}

TEST_CASE("triangular_solve_inplace upper",
          "[triangular_solve][inplace]") {
    Matrix<double, 3, 3> M{{3.0, 1.0, 2.0},
                           {0.0, 4.0, 1.0},
                           {0.0, 0.0, 2.0}};
    Vector<double, 3> x{7.0, 9.0, 2.0};  // starts as b

    auto U = triangular_view<TriangularMode::Upper>(M);
    auto status = triangular_solve_inplace(U, x);

    REQUIRE(status.has_value());
    CHECK(x(0ul) == Catch::Approx(1.0).epsilon(1e-12));
    CHECK(x(1ul) == Catch::Approx(2.0).epsilon(1e-12));
    CHECK(x(2ul) == Catch::Approx(1.0).epsilon(1e-12));
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1x1 system (edge case)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("triangular_solve 1x1", "[triangular_solve][edge]") {
    Matrix<double, 1, 1> M{{5.0}};
    Vector<double, 1> b{10.0};

    auto L = triangular_view<TriangularMode::Lower>(M);
    auto result = triangular_solve(L, b);
    REQUIRE(result.has_value());
    CHECK((*result)(0ul) == Catch::Approx(2.0).epsilon(1e-12));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Larger system (4x4) to verify correctness
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("triangular_solve upper 4x4", "[triangular_solve][upper]") {
    // U = [1  2  3  4]    x = [1, 1, 1, 1]
    //     [0  2  1  1]    b[3] = 5*1 = 5
    //     [0  0  3  2]    b[2] = 3*1 + 2*1 = 5
    //     [0  0  0  5]    b[1] = 2*1 + 1*1 + 1*1 = 4
    //                     b[0] = 1*1 + 2*1 + 3*1 + 4*1 = 10
    Matrix<double, 4, 4> M{
        {1.0, 2.0, 3.0, 4.0},
        {0.0, 2.0, 1.0, 1.0},
        {0.0, 0.0, 3.0, 2.0},
        {0.0, 0.0, 0.0, 5.0}};
    Vector<double, 4> b{10.0, 4.0, 5.0, 5.0};

    auto U = triangular_view<TriangularMode::Upper>(M);
    auto result = triangular_solve(U, b);

    REQUIRE(result.has_value());
    for (index_type i = 0; i < 4; ++i) {
        CHECK((*result)(i) == Catch::Approx(1.0).epsilon(1e-12));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Verify solve with non-trivial solution
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("triangular_solve lower non-trivial solution",
          "[triangular_solve][lower]") {
    // L = [1   0   0  ]    x = [2, -1, 3]
    //     [-2  4   0  ]    b[0] = 1*2 = 2
    //     [1  -1   2  ]    b[1] = -2*2 + 4*(-1) = -8
    //                      b[2] = 1*2 + (-1)*(-1) + 2*3 = 9
    Matrix<double, 3, 3> M{{1.0, 0.0, 0.0},
                           {-2.0, 4.0, 0.0},
                           {1.0, -1.0, 2.0}};
    Vector<double, 3> b{2.0, -8.0, 9.0};

    auto L = triangular_view<TriangularMode::Lower>(M);
    auto result = triangular_solve(L, b);

    REQUIRE(result.has_value());
    CHECK((*result)(0ul) == Catch::Approx(2.0).epsilon(1e-12));
    CHECK((*result)(1ul) == Catch::Approx(-1.0).epsilon(1e-12));
    CHECK((*result)(2ul) == Catch::Approx(3.0).epsilon(1e-12));
}
