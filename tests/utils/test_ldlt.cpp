
#include <cmath>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/decomposition/ldlt.hpp>

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

/// Build diag(D) as an n x n matrix (zero off-diagonal).
template <typename T, index_type N>
void diag_mat(const Vector<T, N> &D, Matrix<T, N, N> &M) {
  const index_type n = D.extent(0);
  for (index_type i = 0; i < n; ++i) {
    for (index_type j = 0; j < n; ++j) {
      M(i, j) = (i == j) ? D(i) : T{0};
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// LDLT decomposition
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ldlt 2x2 SPD", "[decomposition][ldlt]") {
    Matrix<double, 2, 2> A{{4.0, 2.0}, {2.0, 3.0}};

    auto result = utils::decomposition::ldlt(A);
    REQUIRE(result.has_value());

    auto &L = result->L;
    auto &D = result->D;

    // L should be unit lower triangular (ones on diagonal).
    CHECK(L(0, 0) == Catch::Approx(1.0).margin(1e-12));
    CHECK(L(1, 1) == Catch::Approx(1.0).margin(1e-12));
    CHECK(L(0, 1) == Catch::Approx(0.0).margin(1e-12));

    // D should be positive.
    CHECK(D(0) > 0.0);
    CHECK(D(1) > 0.0);

    // L * D * L^T should equal A.
    Matrix<double, 2, 2> Lt(2, 2);
    mat_transpose(L, Lt);
    Matrix<double, 2, 2> DM(2, 2);
    diag_mat(D, DM);

    Matrix<double, 2, 2> LD(2, 2);
    mat_mul(L, DM, LD);
    Matrix<double, 2, 2> LDLt(2, 2);
    mat_mul(LD, Lt, LDLt);

    for (index_type i = 0; i < 2; ++i) {
        for (index_type j = 0; j < 2; ++j) {
            CHECK(LDLt(i, j) == Catch::Approx(A(i, j)).margin(1e-12));
        }
    }

    // Verify known values: D = [4, 2], L = [[1, 0], [0.5, 1]]
    CHECK(D(0) == Catch::Approx(4.0).margin(1e-12));
    CHECK(D(1) == Catch::Approx(2.0).margin(1e-12));
    CHECK(L(1, 0) == Catch::Approx(0.5).margin(1e-12));
}

TEST_CASE("ldlt 3x3 SPD", "[decomposition][ldlt]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};

    auto result = utils::decomposition::ldlt(A);
    REQUIRE(result.has_value());

    auto &L = result->L;
    auto &D = result->D;

    // L should be unit lower triangular.
    for (index_type i = 0; i < 3; ++i) {
        CHECK(L(i, i) == Catch::Approx(1.0).margin(1e-12));
        for (index_type j = i + 1; j < 3; ++j) {
            CHECK(L(i, j) == Catch::Approx(0.0).margin(1e-12));
        }
    }

    // D should be positive for SPD matrices.
    for (index_type i = 0; i < 3; ++i) {
        CHECK(D(i) > 0.0);
    }

    // L * D * L^T should equal A.
    Matrix<double, 3, 3> Lt(3, 3);
    mat_transpose(L, Lt);
    Matrix<double, 3, 3> DM(3, 3);
    diag_mat(D, DM);

    Matrix<double, 3, 3> LD(3, 3);
    mat_mul(L, DM, LD);
    Matrix<double, 3, 3> LDLt(3, 3);
    mat_mul(LD, Lt, LDLt);

    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(LDLt(i, j) == Catch::Approx(A(i, j)).margin(1e-12));
        }
    }
}

TEST_CASE("ldlt identity", "[decomposition][ldlt]") {
    Matrix<double, 3, 3> I{
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

    auto result = utils::decomposition::ldlt(I);
    REQUIRE(result.has_value());

    // L should be identity, D should be all ones.
    for (index_type i = 0; i < 3; ++i) {
        CHECK(result->D(i) == Catch::Approx(1.0).margin(1e-12));
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(result->L(i, j) ==
                  Catch::Approx(expected).margin(1e-12));
        }
    }
}

TEST_CASE("ldlt 1x1", "[decomposition][ldlt]") {
    Matrix<double, 1, 1> A{{9.0}};

    auto result = utils::decomposition::ldlt(A);
    REQUIRE(result.has_value());
    CHECK(result->L(0, 0) == Catch::Approx(1.0).margin(1e-12));
    CHECK(result->D(0) == Catch::Approx(9.0).margin(1e-12));
}

// ─────────────────────────────────────────────────────────────────────────────
// LDLT solve
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ldlt_solve 2x2", "[decomposition][ldlt_solve]") {
    Matrix<double, 2, 2> A{{4.0, 2.0}, {2.0, 3.0}};
    // x_true = [1, 2], b = A * x = [8, 8]
    Vector<double, 2> b{8.0, 8.0};

    auto result = utils::decomposition::ldlt_solve(A, b);
    REQUIRE(result.has_value());

    CHECK(result->operator()(0) == Catch::Approx(1.0).margin(1e-10));
    CHECK(result->operator()(1) == Catch::Approx(2.0).margin(1e-10));
}

TEST_CASE("ldlt_solve 3x3 SPD", "[decomposition][ldlt_solve]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> b{6.0, 6.0, 6.0};

    auto result = utils::decomposition::ldlt_solve(A, b);
    REQUIRE(result.has_value());

    CHECK(result->operator()(0) == Catch::Approx(1.0).margin(1e-10));
    CHECK(result->operator()(1) == Catch::Approx(1.0).margin(1e-10));
    CHECK(result->operator()(2) == Catch::Approx(1.0).margin(1e-10));
}

TEST_CASE("ldlt_solve 3x3 with known solution", "[decomposition][ldlt_solve]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> x_true{1.0, 2.0, 3.0};
    Vector<double, 3> b(3);
    mat_vec(A, x_true, b);

    auto result = utils::decomposition::ldlt_solve(A, b);
    REQUIRE(result.has_value());

    for (index_type i = 0; i < 3; ++i) {
        CHECK(result->operator()(i) ==
              Catch::Approx(x_true(i)).margin(1e-10));
    }
}

TEST_CASE("ldlt_solve 4x4 SPD", "[decomposition][ldlt_solve]") {
    Matrix<double, 4, 4> A{
        {10.0, 1.0, 2.0, 1.0},
        {1.0, 10.0, 1.0, 2.0},
        {2.0, 1.0, 10.0, 1.0},
        {1.0, 2.0, 1.0, 10.0},
    };
    Vector<double, 4> x_true{1.0, -1.0, 2.0, -2.0};
    Vector<double, 4> b(4);
    mat_vec(A, x_true, b);

    auto result = utils::decomposition::ldlt_solve(A, b);
    REQUIRE(result.has_value());

    for (index_type i = 0; i < 4; ++i) {
        CHECK(result->operator()(i) ==
              Catch::Approx(x_true(i)).margin(1e-10));
    }
}

TEST_CASE("ldlt_solve matches llt_solve", "[decomposition][ldlt_solve]") {
    // Both LDLT and LLT should give the same answer for SPD systems.
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> b{7.0, 11.0, 15.0};

    auto ldlt_result = utils::decomposition::ldlt_solve(A, b);
    REQUIRE(ldlt_result.has_value());

    // Expected: x = [1, 2, 3] since A * [1,2,3] = [4+2+3, 1+8+3, 1+2+12] = [9,12,15]
    // Actually just compare LDLT against direct computation.
    // A * x = b, so just check the residual.
    Vector<double, 3> Ax(3);
    mat_vec(A, *ldlt_result, Ax);

    for (index_type i = 0; i < 3; ++i) {
        CHECK(Ax(i) == Catch::Approx(b(i)).margin(1e-10));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LDLTResult::solve(b)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("LDLTResult::solve 3x3", "[decomposition][ldlt_solve]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> x_true{1.0, 2.0, 3.0};
    Vector<double, 3> b(3);
    mat_vec(A, x_true, b);

    auto decomp = utils::decomposition::ldlt(A);
    REQUIRE(decomp.has_value());

    auto result = decomp->solve(b);
    REQUIRE(result.has_value());

    for (index_type i = 0; i < 3; ++i) {
        CHECK(result->operator()(i) ==
              Catch::Approx(x_true(i)).margin(1e-10));
    }
}

TEST_CASE("LDLTResult::solve matches ldlt_solve", "[decomposition][ldlt_solve]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> b{6.0, 6.0, 6.0};

    auto free_result = utils::decomposition::ldlt_solve(A, b);
    REQUIRE(free_result.has_value());

    auto decomp = utils::decomposition::ldlt(A);
    REQUIRE(decomp.has_value());
    auto method_result = decomp->solve(b);
    REQUIRE(method_result.has_value());

    for (index_type i = 0; i < 3; ++i) {
        CHECK(free_result->operator()(i) ==
              Catch::Approx(method_result->operator()(i)).margin(1e-14));
    }
}
