
#include <cmath>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/decomposition/llt.hpp>

#include "../catch_include.hpp"

using namespace zipper;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute C = A * B element-by-element.
template <typename T, index_type M, index_type K, index_type N>
void mat_mul(const Matrix<T, M, K> &A, const Matrix<T, K, N> &B,
             Matrix<T, M, N> &C) {
  const index_type m = A.extent(0);
  const index_type k = A.extent(1);
  const index_type n = B.extent(1);
  for (index_type i = 0; i < m; ++i) {
    for (index_type j = 0; j < n; ++j) {
      T sum = T{0};
      for (index_type l = 0; l < k; ++l) {
        sum += A(i, l) * B(l, j);
      }
      C(i, j) = sum;
    }
  }
}

/// Compute B = transpose(A) element-by-element.
template <typename T, index_type M, index_type N>
void mat_transpose(const Matrix<T, M, N> &A, Matrix<T, N, M> &B) {
  const index_type m = A.extent(0);
  const index_type n = A.extent(1);
  for (index_type i = 0; i < m; ++i) {
    for (index_type j = 0; j < n; ++j) {
      B(j, i) = A(i, j);
    }
  }
}

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
// LLT decomposition
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("llt 2x2 SPD", "[decomposition][llt]") {
    // A = [[4, 2], [2, 3]]
    Matrix<double, 2, 2> A{{4.0, 2.0}, {2.0, 3.0}};

    auto result = utils::decomposition::llt(A);
    REQUIRE(result.has_value());

    auto &L = result->L;

    // L should be lower triangular.
    CHECK(L(0, 1) == Catch::Approx(0.0).margin(1e-12));

    // L * L^T should equal A.
    Matrix<double, 2, 2> Lt(2, 2);
    mat_transpose(L, Lt);
    Matrix<double, 2, 2> LLt(2, 2);
    mat_mul(L, Lt, LLt);

    for (index_type i = 0; i < 2; ++i) {
        for (index_type j = 0; j < 2; ++j) {
            CHECK(LLt(i, j) == Catch::Approx(A(i, j)).margin(1e-12));
        }
    }

    // Verify known values: L = [[2, 0], [1, sqrt(2)]]
    CHECK(L(0, 0) == Catch::Approx(2.0).margin(1e-12));
    CHECK(L(1, 0) == Catch::Approx(1.0).margin(1e-12));
    CHECK(L(1, 1) == Catch::Approx(std::sqrt(2.0)).margin(1e-12));
}

TEST_CASE("llt 3x3 SPD", "[decomposition][llt]") {
    // A = [[4, 1, 1], [1, 4, 1], [1, 1, 4]]  (diagonally dominant SPD)
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};

    auto result = utils::decomposition::llt(A);
    REQUIRE(result.has_value());

    auto &L = result->L;

    // L should be lower triangular.
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = i + 1; j < 3; ++j) {
            CHECK(L(i, j) == Catch::Approx(0.0).margin(1e-12));
        }
    }

    // Diagonal should be positive.
    for (index_type i = 0; i < 3; ++i) {
        CHECK(L(i, i) > 0.0);
    }

    // L * L^T should equal A.
    Matrix<double, 3, 3> Lt(3, 3);
    mat_transpose(L, Lt);
    Matrix<double, 3, 3> LLt(3, 3);
    mat_mul(L, Lt, LLt);

    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(LLt(i, j) == Catch::Approx(A(i, j)).margin(1e-12));
        }
    }
}

TEST_CASE("llt identity", "[decomposition][llt]") {
    // L of identity should be identity.
    Matrix<double, 3, 3> I{
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

    auto result = utils::decomposition::llt(I);
    REQUIRE(result.has_value());

    auto &L = result->L;
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(L(i, j) == Catch::Approx(expected).margin(1e-12));
        }
    }
}

TEST_CASE("llt non-positive-definite fails", "[decomposition][llt]") {
    // A = [[1, 2], [2, 1]] has eigenvalues -1, 3 — not positive definite.
    Matrix<double, 2, 2> A{{1.0, 2.0}, {2.0, 1.0}};

    auto result = utils::decomposition::llt(A);
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().kind == utils::solver::SolverError::Kind::breakdown);
}

TEST_CASE("llt 1x1", "[decomposition][llt]") {
    Matrix<double, 1, 1> A{{9.0}};

    auto result = utils::decomposition::llt(A);
    REQUIRE(result.has_value());
    CHECK(result->L(0, 0) == Catch::Approx(3.0).margin(1e-12));
}

// ─────────────────────────────────────────────────────────────────────────────
// LLT solve
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("llt_solve 2x2", "[decomposition][llt_solve]") {
    Matrix<double, 2, 2> A{{4.0, 2.0}, {2.0, 3.0}};
    // x_true = [1, 2], b = A * x = [8, 8]
    Vector<double, 2> b{8.0, 8.0};

    auto result = utils::decomposition::llt_solve(A, b);
    REQUIRE(result.has_value());

    CHECK(result->operator()(0) == Catch::Approx(1.0).margin(1e-10));
    CHECK(result->operator()(1) == Catch::Approx(2.0).margin(1e-10));
}

TEST_CASE("llt_solve 3x3 SPD", "[decomposition][llt_solve]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> b{6.0, 6.0, 6.0};

    auto result = utils::decomposition::llt_solve(A, b);
    REQUIRE(result.has_value());

    CHECK(result->operator()(0) == Catch::Approx(1.0).margin(1e-10));
    CHECK(result->operator()(1) == Catch::Approx(1.0).margin(1e-10));
    CHECK(result->operator()(2) == Catch::Approx(1.0).margin(1e-10));
}

TEST_CASE("llt_solve 3x3 with known solution", "[decomposition][llt_solve]") {
    // A = [[4, 1, 1], [1, 4, 1], [1, 1, 4]], x_true = [1, 2, 3]
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> x_true{1.0, 2.0, 3.0};
    Vector<double, 3> b(3);
    mat_vec(A, x_true, b);

    auto result = utils::decomposition::llt_solve(A, b);
    REQUIRE(result.has_value());

    for (index_type i = 0; i < 3; ++i) {
        CHECK(result->operator()(i) ==
              Catch::Approx(x_true(i)).margin(1e-10));
    }
}

TEST_CASE("llt_solve 4x4 SPD", "[decomposition][llt_solve]") {
    // 4x4 diagonally dominant SPD matrix.
    Matrix<double, 4, 4> A{
        {10.0, 1.0, 2.0, 1.0},
        {1.0, 10.0, 1.0, 2.0},
        {2.0, 1.0, 10.0, 1.0},
        {1.0, 2.0, 1.0, 10.0},
    };
    Vector<double, 4> x_true{1.0, -1.0, 2.0, -2.0};
    Vector<double, 4> b(4);
    mat_vec(A, x_true, b);

    auto result = utils::decomposition::llt_solve(A, b);
    REQUIRE(result.has_value());

    for (index_type i = 0; i < 4; ++i) {
        CHECK(result->operator()(i) ==
              Catch::Approx(x_true(i)).margin(1e-10));
    }
}

TEST_CASE("llt_solve non-SPD fails", "[decomposition][llt_solve]") {
    Matrix<double, 2, 2> A{{1.0, 2.0}, {2.0, 1.0}};
    Vector<double, 2> b{1.0, 1.0};

    auto result = utils::decomposition::llt_solve(A, b);
    REQUIRE_FALSE(result.has_value());
}

// ─────────────────────────────────────────────────────────────────────────────
// LLTResult::solve(b)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("LLTResult::solve 3x3", "[decomposition][llt_solve]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> x_true{1.0, 2.0, 3.0};
    Vector<double, 3> b(3);
    mat_vec(A, x_true, b);

    auto decomp = utils::decomposition::llt(A);
    REQUIRE(decomp.has_value());

    auto result = decomp->solve(b);
    REQUIRE(result.has_value());

    for (index_type i = 0; i < 3; ++i) {
        CHECK(result->operator()(i) ==
              Catch::Approx(x_true(i)).margin(1e-10));
    }
}

TEST_CASE("LLTResult::solve matches llt_solve", "[decomposition][llt_solve]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> b{6.0, 6.0, 6.0};

    // Compare free function and method results.
    auto free_result = utils::decomposition::llt_solve(A, b);
    REQUIRE(free_result.has_value());

    auto decomp = utils::decomposition::llt(A);
    REQUIRE(decomp.has_value());
    auto method_result = decomp->solve(b);
    REQUIRE(method_result.has_value());

    for (index_type i = 0; i < 3; ++i) {
        CHECK(free_result->operator()(i) ==
              Catch::Approx(method_result->operator()(i)).margin(1e-14));
    }
}
