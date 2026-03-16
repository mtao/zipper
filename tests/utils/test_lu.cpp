
#include <cmath>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/decomposition/lu.hpp>

#include "../catch_include.hpp"

using namespace zipper;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute y = A * x element-by-element (matrix-vector product).
template <typename T, index_type M, index_type N>
void mat_vec(const Matrix<T, M, N> &A, const Vector<T, N> &x,
             Vector<T, M> &y) {
  const index_type m = A.extent(0);
  const index_type n = A.extent(1);
  for (index_type i = 0; i < m; ++i) {
    T sum = T{0};
    for (index_type j = 0; j < n; ++j) {
      sum += A(i, j) * x(j);
    }
    y(i) = sum;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// PLU decomposition
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("plu 2x2", "[decomposition][plu]") {
    Matrix<double, 2, 2> A{{2.0, 1.0}, {4.0, 3.0}};

    auto result = utils::decomposition::plu(A);
    REQUIRE(result.has_value());

    auto &LU = result->LU;
    auto &perm = result->perm;

    // Verify: P*A = L*U by reconstructing.
    const index_type n = 2;

    // Extract L and U from combined matrix.
    Matrix<double, 2, 2> L(2, 2);
    Matrix<double, 2, 2> U(2, 2);
    for (index_type i = 0; i < n; ++i) {
        for (index_type j = 0; j < n; ++j) {
            if (i > j) {
                L(i, j) = LU(i, j);
                U(i, j) = 0.0;
            } else if (i == j) {
                L(i, j) = 1.0;
                U(i, j) = LU(i, j);
            } else {
                L(i, j) = 0.0;
                U(i, j) = LU(i, j);
            }
        }
    }

    // L*U should equal P*A.
    for (index_type i = 0; i < n; ++i) {
        for (index_type j = 0; j < n; ++j) {
            double lu_ij = 0.0;
            for (index_type k = 0; k < n; ++k) {
                lu_ij += L(i, k) * U(k, j);
            }
            CHECK(lu_ij == Catch::Approx(A(perm[i], j)).margin(1e-12));
        }
    }
}

TEST_CASE("plu 3x3", "[decomposition][plu]") {
    Matrix<double, 3, 3> A{
        {2.0, 1.0, 1.0}, {4.0, 3.0, 3.0}, {8.0, 7.0, 9.0}};

    auto result = utils::decomposition::plu(A);
    REQUIRE(result.has_value());

    auto &LU = result->LU;
    auto &perm = result->perm;
    const index_type n = 3;

    // Reconstruct and verify P*A = L*U.
    Matrix<double, 3, 3> L(3, 3);
    Matrix<double, 3, 3> U(3, 3);
    for (index_type i = 0; i < n; ++i) {
        for (index_type j = 0; j < n; ++j) {
            if (i > j) {
                L(i, j) = LU(i, j);
                U(i, j) = 0.0;
            } else if (i == j) {
                L(i, j) = 1.0;
                U(i, j) = LU(i, j);
            } else {
                L(i, j) = 0.0;
                U(i, j) = LU(i, j);
            }
        }
    }

    for (index_type i = 0; i < n; ++i) {
        for (index_type j = 0; j < n; ++j) {
            double lu_ij = 0.0;
            for (index_type k = 0; k < n; ++k) {
                lu_ij += L(i, k) * U(k, j);
            }
            CHECK(lu_ij == Catch::Approx(A(perm[i], j)).margin(1e-12));
        }
    }
}

TEST_CASE("plu identity", "[decomposition][plu]") {
    Matrix<double, 3, 3> I{
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

    auto result = utils::decomposition::plu(I);
    REQUIRE(result.has_value());

    // Determinant of identity is 1.
    CHECK(result->determinant() == Catch::Approx(1.0).margin(1e-12));
}

TEST_CASE("plu singular fails", "[decomposition][plu]") {
    // Singular matrix: row 2 = 2 * row 1.
    Matrix<double, 2, 2> A{{1.0, 2.0}, {2.0, 4.0}};

    auto result = utils::decomposition::plu(A);
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().kind == utils::solver::SolverError::Kind::breakdown);
}

TEST_CASE("plu 1x1", "[decomposition][plu]") {
    Matrix<double, 1, 1> A{{5.0}};

    auto result = utils::decomposition::plu(A);
    REQUIRE(result.has_value());
    CHECK(result->LU(0, 0) == Catch::Approx(5.0).margin(1e-12));
    CHECK(result->determinant() == Catch::Approx(5.0).margin(1e-12));
}

// ─────────────────────────────────────────────────────────────────────────────
// PLU determinant
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("plu determinant 2x2", "[decomposition][plu]") {
    Matrix<double, 2, 2> A{{3.0, 8.0}, {4.0, 6.0}};
    // det(A) = 3*6 - 8*4 = 18 - 32 = -14
    auto result = utils::decomposition::plu(A);
    REQUIRE(result.has_value());
    CHECK(result->determinant() == Catch::Approx(-14.0).margin(1e-10));
}

TEST_CASE("plu determinant 3x3", "[decomposition][plu]") {
    Matrix<double, 3, 3> A{
        {6.0, 1.0, 1.0}, {4.0, -2.0, 5.0}, {2.0, 8.0, 7.0}};
    // det(A) = 6*(-2*7 - 5*8) - 1*(4*7 - 5*2) + 1*(4*8 - (-2)*2)
    //        = 6*(-14-40) - 1*(28-10) + 1*(32+4)
    //        = 6*(-54) - 18 + 36 = -324 - 18 + 36 = -306
    auto result = utils::decomposition::plu(A);
    REQUIRE(result.has_value());
    CHECK(result->determinant() == Catch::Approx(-306.0).margin(1e-8));
}

// ─────────────────────────────────────────────────────────────────────────────
// PLU solve
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("plu_solve 2x2", "[decomposition][plu_solve]") {
    Matrix<double, 2, 2> A{{2.0, 1.0}, {4.0, 3.0}};
    // x_true = [1, 2], b = A * x = [4, 10]
    Vector<double, 2> b{4.0, 10.0};

    auto result = utils::decomposition::plu_solve(A, b);
    REQUIRE(result.has_value());

    CHECK(result->operator()(0) == Catch::Approx(1.0).margin(1e-10));
    CHECK(result->operator()(1) == Catch::Approx(2.0).margin(1e-10));
}

TEST_CASE("plu_solve 3x3", "[decomposition][plu_solve]") {
    Matrix<double, 3, 3> A{
        {2.0, 1.0, 1.0}, {4.0, 3.0, 3.0}, {8.0, 7.0, 9.0}};
    Vector<double, 3> x_true{1.0, 2.0, 3.0};
    Vector<double, 3> b(3);
    mat_vec(A, x_true, b);

    auto result = utils::decomposition::plu_solve(A, b);
    REQUIRE(result.has_value());

    for (index_type i = 0; i < 3; ++i) {
        CHECK(result->operator()(i) ==
              Catch::Approx(x_true(i)).margin(1e-10));
    }
}

TEST_CASE("plu_solve 3x3 non-symmetric", "[decomposition][plu_solve]") {
    // Non-symmetric matrix: PLU handles general matrices, unlike LLT/LDLT.
    Matrix<double, 3, 3> A{
        {3.0, 1.0, 0.0}, {0.0, 3.0, 1.0}, {0.0, 0.0, 3.0}};
    Vector<double, 3> b{4.0, 4.0, 3.0};

    auto result = utils::decomposition::plu_solve(A, b);
    REQUIRE(result.has_value());

    CHECK(result->operator()(0) == Catch::Approx(1.0).margin(1e-10));
    CHECK(result->operator()(1) == Catch::Approx(1.0).margin(1e-10));
    CHECK(result->operator()(2) == Catch::Approx(1.0).margin(1e-10));
}

TEST_CASE("plu_solve 4x4", "[decomposition][plu_solve]") {
    Matrix<double, 4, 4> A{
        {10.0, 1.0, 2.0, 1.0},
        {1.0, 10.0, 1.0, 2.0},
        {2.0, 1.0, 10.0, 1.0},
        {1.0, 2.0, 1.0, 10.0},
    };
    Vector<double, 4> x_true{1.0, -1.0, 2.0, -2.0};
    Vector<double, 4> b(4);
    mat_vec(A, x_true, b);

    auto result = utils::decomposition::plu_solve(A, b);
    REQUIRE(result.has_value());

    for (index_type i = 0; i < 4; ++i) {
        CHECK(result->operator()(i) ==
              Catch::Approx(x_true(i)).margin(1e-10));
    }
}

TEST_CASE("plu_solve singular fails", "[decomposition][plu_solve]") {
    Matrix<double, 2, 2> A{{1.0, 2.0}, {2.0, 4.0}};
    Vector<double, 2> b{1.0, 2.0};

    auto result = utils::decomposition::plu_solve(A, b);
    REQUIRE_FALSE(result.has_value());
}

// ─────────────────────────────────────────────────────────────────────────────
// PLUResult::solve(b) — re-use the factorisation
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("PLUResult::solve reuses factorisation", "[decomposition][plu_solve]") {
    Matrix<double, 3, 3> A{
        {2.0, 1.0, 1.0}, {4.0, 3.0, 3.0}, {8.0, 7.0, 9.0}};

    auto decomp = utils::decomposition::plu(A);
    REQUIRE(decomp.has_value());

    // Solve for two different right-hand sides using the same factorisation.
    Vector<double, 3> x1_true{1.0, 0.0, 0.0};
    Vector<double, 3> b1(3);
    mat_vec(A, x1_true, b1);

    auto result1 = decomp->solve(b1);
    REQUIRE(result1.has_value());
    for (index_type i = 0; i < 3; ++i) {
        CHECK(result1->operator()(i) ==
              Catch::Approx(x1_true(i)).margin(1e-10));
    }

    Vector<double, 3> x2_true{0.0, 0.0, 1.0};
    Vector<double, 3> b2(3);
    mat_vec(A, x2_true, b2);

    auto result2 = decomp->solve(b2);
    REQUIRE(result2.has_value());
    for (index_type i = 0; i < 3; ++i) {
        CHECK(result2->operator()(i) ==
              Catch::Approx(x2_true(i)).margin(1e-10));
    }
}

TEST_CASE("PLUResult::solve matches plu_solve", "[decomposition][plu_solve]") {
    Matrix<double, 3, 3> A{
        {2.0, 1.0, 1.0}, {4.0, 3.0, 3.0}, {8.0, 7.0, 9.0}};
    Vector<double, 3> b{3.0, 7.0, 15.0};

    auto free_result = utils::decomposition::plu_solve(A, b);
    REQUIRE(free_result.has_value());

    auto decomp = utils::decomposition::plu(A);
    REQUIRE(decomp.has_value());
    auto method_result = decomp->solve(b);
    REQUIRE(method_result.has_value());

    for (index_type i = 0; i < 3; ++i) {
        CHECK(free_result->operator()(i) ==
              Catch::Approx(method_result->operator()(i)).margin(1e-14));
    }
}
