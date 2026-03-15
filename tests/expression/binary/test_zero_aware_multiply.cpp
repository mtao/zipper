// Tests for zero-aware matrix/vector multiplication (Phase 4):
// - MatrixProduct with triangular views (known zeros skip inner loop iterations)
// - MatrixProduct with Identity (single-element inner loop)
// - MatrixVectorProduct with triangular matrix * unit vector
// - MatrixVectorProduct with triangular matrix * dense vector
// - Correctness verified against naive dense computation

#include "../../catch_include.hpp"

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/Form.hpp>
#include <zipper/MatrixBase.hxx>
#include <zipper/VectorBase.hxx>
#include <zipper/FormBase.hxx>
#include <zipper/expression/binary/MatrixProduct.hpp>
#include <zipper/expression/binary/MatrixVectorProduct.hpp>
#include <zipper/expression/detail/ExpressionTraits.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/Unit.hpp>
#include <zipper/expression/unary/TriangularView.hpp>

using namespace zipper;
using namespace zipper::expression;

// ═══════════════════════════════════════════════════════════════════════════════
// Helper: dense reference multiplication for correctness checks
// ═══════════════════════════════════════════════════════════════════════════════

template <typename M1, typename M2>
double dense_matmul(const M1& a, const M2& b, index_type i, index_type j) {
    double v = 0;
    for (index_type k = 0; k < a.extent(1); ++k) {
        v += a(i, k) * b(k, j);
    }
    return v;
}

template <typename M, typename V>
double dense_matvec(const M& a, const V& x, index_type i) {
    double v = 0;
    for (index_type k = 0; k < a.extent(1); ++k) {
        v += a(i, k) * x(k);
    }
    return v;
}

// ═══════════════════════════════════════════════════════════════════════════════
// MatrixProduct: Identity * Matrix
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Zero-aware matmul: Identity * Matrix",
          "[zero_aware_multiply][matrix_product][identity]") {
    Matrix<double, 4, 4> A;
    A = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

    MatrixBase I = nullary::Identity<double, 4, 4>{};

    // I * A should equal A
    auto result = I * A;

    for (index_type i = 0; i < 4; ++i) {
        for (index_type j = 0; j < 4; ++j) {
            CHECK(result(i, j) == Catch::Approx(A(i, j)));
        }
    }
}

TEST_CASE("Zero-aware matmul: Matrix * Identity",
          "[zero_aware_multiply][matrix_product][identity]") {
    Matrix<double, 4, 4> A;
    A = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

    MatrixBase I = nullary::Identity<double, 4, 4>{};

    // A * I should equal A
    auto result = A * I;

    for (index_type i = 0; i < 4; ++i) {
        for (index_type j = 0; j < 4; ++j) {
            CHECK(result(i, j) == Catch::Approx(A(i, j)));
        }
    }
}

TEST_CASE("Zero-aware matmul: Identity * Identity",
          "[zero_aware_multiply][matrix_product][identity]") {
    MatrixBase I = nullary::Identity<double, 3, 3>{};

    auto result = I * I;

    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(result(i, j) == Catch::Approx(expected));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MatrixProduct: Triangular * Triangular
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Zero-aware matmul: Lower * Lower",
          "[zero_aware_multiply][matrix_product][triangular]") {
    Matrix<double, 4, 4> A;
    A = {{2, 0, 0, 0}, {1, 3, 0, 0}, {4, 2, 1, 0}, {3, 1, 2, 5}};

    auto L = MatrixBase(triangular_view<TriangularMode::Lower>(A));

    // L * L
    auto result = L * L;

    for (index_type i = 0; i < 4; ++i) {
        for (index_type j = 0; j < 4; ++j) {
            CHECK(result(i, j) == Catch::Approx(dense_matmul(L, L, i, j)));
        }
    }
}

TEST_CASE("Zero-aware matmul: Upper * Upper",
          "[zero_aware_multiply][matrix_product][triangular]") {
    Matrix<double, 4, 4> A;
    A = {{2, 1, 4, 3}, {0, 3, 2, 1}, {0, 0, 1, 2}, {0, 0, 0, 5}};

    auto U = MatrixBase(triangular_view<TriangularMode::Upper>(A));

    auto result = U * U;

    for (index_type i = 0; i < 4; ++i) {
        for (index_type j = 0; j < 4; ++j) {
            CHECK(result(i, j) == Catch::Approx(dense_matmul(U, U, i, j)));
        }
    }
}

TEST_CASE("Zero-aware matmul: Lower * Upper",
          "[zero_aware_multiply][matrix_product][triangular]") {
    Matrix<double, 4, 4> A;
    A = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

    auto L = MatrixBase(triangular_view<TriangularMode::Lower>(A));
    auto U = MatrixBase(triangular_view<TriangularMode::Upper>(A));

    auto result = L * U;

    for (index_type i = 0; i < 4; ++i) {
        for (index_type j = 0; j < 4; ++j) {
            CHECK(result(i, j) == Catch::Approx(dense_matmul(L, U, i, j)));
        }
    }
}

TEST_CASE("Zero-aware matmul: Upper * Lower",
          "[zero_aware_multiply][matrix_product][triangular]") {
    Matrix<double, 4, 4> A;
    A = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

    auto L = MatrixBase(triangular_view<TriangularMode::Lower>(A));
    auto U = MatrixBase(triangular_view<TriangularMode::Upper>(A));

    auto result = U * L;

    for (index_type i = 0; i < 4; ++i) {
        for (index_type j = 0; j < 4; ++j) {
            CHECK(result(i, j) == Catch::Approx(dense_matmul(U, L, i, j)));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MatrixProduct: Triangular * Dense
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Zero-aware matmul: Lower * Dense",
          "[zero_aware_multiply][matrix_product][triangular]") {
    Matrix<double, 3, 3> A;
    A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    Matrix<double, 3, 3> B;
    B = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};

    auto L = MatrixBase(triangular_view<TriangularMode::Lower>(A));

    auto result = L * B;

    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(result(i, j) == Catch::Approx(dense_matmul(L, B, i, j)));
        }
    }
}

TEST_CASE("Zero-aware matmul: Dense * Upper",
          "[zero_aware_multiply][matrix_product][triangular]") {
    Matrix<double, 3, 3> A;
    A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    Matrix<double, 3, 3> B;
    B = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};

    auto U = MatrixBase(triangular_view<TriangularMode::Upper>(B));

    auto result = A * U;

    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(result(i, j) == Catch::Approx(dense_matmul(A, U, i, j)));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MatrixVectorProduct: Triangular * Vector
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Zero-aware matvec: Lower * vector",
          "[zero_aware_multiply][matvec][triangular]") {
    Matrix<double, 4, 4> A;
    A = {{2, 0, 0, 0}, {1, 3, 0, 0}, {4, 2, 1, 0}, {3, 1, 2, 5}};

    Vector<double, 4> x;
    x = {1.0, 2.0, 3.0, 4.0};

    auto L = MatrixBase(triangular_view<TriangularMode::Lower>(A));

    auto result = L * x;

    for (index_type i = 0; i < 4; ++i) {
        CHECK(result(i) == Catch::Approx(dense_matvec(L, x, i)));
    }
}

TEST_CASE("Zero-aware matvec: Upper * vector",
          "[zero_aware_multiply][matvec][triangular]") {
    Matrix<double, 4, 4> A;
    A = {{2, 1, 4, 3}, {0, 3, 2, 1}, {0, 0, 1, 2}, {0, 0, 0, 5}};

    Vector<double, 4> x;
    x = {1.0, 2.0, 3.0, 4.0};

    auto U = MatrixBase(triangular_view<TriangularMode::Upper>(A));

    auto result = U * x;

    for (index_type i = 0; i < 4; ++i) {
        CHECK(result(i) == Catch::Approx(dense_matvec(U, x, i)));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MatrixVectorProduct: Matrix * Unit vector
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Zero-aware matvec: Dense * unit vector",
          "[zero_aware_multiply][matvec][unit]") {
    Matrix<double, 3, 3> A;
    A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    // e1 = [0, 1, 0]
    auto e1 = VectorBase(nullary::Unit<double, 3, index_type>(1));

    auto result = A * e1;

    // A * e1 should be column 1 of A
    CHECK(result(0) == Catch::Approx(2.0));
    CHECK(result(1) == Catch::Approx(5.0));
    CHECK(result(2) == Catch::Approx(8.0));
}

TEST_CASE("Zero-aware matvec: Identity * unit vector",
          "[zero_aware_multiply][matvec][identity][unit]") {
    MatrixBase I = nullary::Identity<double, 4, 4>{};

    auto e2 = VectorBase(nullary::Unit<double, 4, index_type>(2));

    // I * e2 = e2
    auto result = I * e2;

    CHECK(result(0) == Catch::Approx(0.0));
    CHECK(result(1) == Catch::Approx(0.0));
    CHECK(result(2) == Catch::Approx(1.0));
    CHECK(result(3) == Catch::Approx(0.0));
}

// ═══════════════════════════════════════════════════════════════════════════════
// MatrixVectorProduct: Triangular * Unit vector (both have known zeros)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Zero-aware matvec: Lower * unit vector",
          "[zero_aware_multiply][matvec][triangular][unit]") {
    Matrix<double, 4, 4> A;
    A = {{2, 0, 0, 0}, {1, 3, 0, 0}, {4, 2, 1, 0}, {3, 1, 2, 5}};

    auto L = MatrixBase(triangular_view<TriangularMode::Lower>(A));

    // e0 = [1, 0, 0, 0]
    auto e0 = VectorBase(nullary::Unit<double, 4, index_type>(0));

    auto result = L * e0;

    // L * e0 = first column of L = [2, 1, 4, 3]
    CHECK(result(0) == Catch::Approx(2.0));
    CHECK(result(1) == Catch::Approx(1.0));
    CHECK(result(2) == Catch::Approx(4.0));
    CHECK(result(3) == Catch::Approx(3.0));
}

TEST_CASE("Zero-aware matvec: Upper * unit vector (last)",
          "[zero_aware_multiply][matvec][triangular][unit]") {
    Matrix<double, 4, 4> A;
    A = {{2, 1, 4, 3}, {0, 3, 2, 1}, {0, 0, 1, 2}, {0, 0, 0, 5}};

    auto U = MatrixBase(triangular_view<TriangularMode::Upper>(A));

    // e3 = [0, 0, 0, 1]
    auto e3 = VectorBase(nullary::Unit<double, 4, index_type>(3));

    auto result = U * e3;

    // U * e3 = last column of U = [3, 1, 2, 5]
    CHECK(result(0) == Catch::Approx(3.0));
    CHECK(result(1) == Catch::Approx(1.0));
    CHECK(result(2) == Catch::Approx(2.0));
    CHECK(result(3) == Catch::Approx(5.0));
}

// ═══════════════════════════════════════════════════════════════════════════════
// MatrixVectorProduct: Identity * dense vector
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Zero-aware matvec: Identity * dense vector",
          "[zero_aware_multiply][matvec][identity]") {
    MatrixBase I = nullary::Identity<double, 4, 4>{};

    Vector<double, 4> x;
    x = {10.0, 20.0, 30.0, 40.0};

    auto result = I * x;

    CHECK(result(0) == Catch::Approx(10.0));
    CHECK(result(1) == Catch::Approx(20.0));
    CHECK(result(2) == Catch::Approx(30.0));
    CHECK(result(3) == Catch::Approx(40.0));
}

// ═══════════════════════════════════════════════════════════════════════════════
// MatrixProduct: StrictlyTriangular variants
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Zero-aware matmul: StrictlyLower * StrictlyUpper",
          "[zero_aware_multiply][matrix_product][strictly_triangular]") {
    Matrix<double, 3, 3> A;
    A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    auto SL = MatrixBase(triangular_view<TriangularMode::StrictlyLower>(A));
    auto SU = MatrixBase(triangular_view<TriangularMode::StrictlyUpper>(A));

    auto result = SL * SU;

    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(result(i, j) == Catch::Approx(dense_matmul(SL, SU, i, j)));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Assignment: zero-aware product result assigned to owning type
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Zero-aware matmul: result assigns to owning Matrix",
          "[zero_aware_multiply][assignment]") {
    Matrix<double, 3, 3> A;
    A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    MatrixBase I = nullary::Identity<double, 3, 3>{};

    Matrix<double, 3, 3> result;
    result = I * A;

    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(result(i, j) == Catch::Approx(A(i, j)));
        }
    }
}

TEST_CASE("Zero-aware matvec: result assigns to owning Vector",
          "[zero_aware_multiply][assignment]") {
    Matrix<double, 3, 3> A;
    A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    auto L = MatrixBase(triangular_view<TriangularMode::Lower>(A));

    Vector<double, 3> x;
    x = {1.0, 1.0, 1.0};

    Vector<double, 3> result;
    result = L * x;

    for (index_type i = 0; i < 3; ++i) {
        CHECK(result(i) == Catch::Approx(dense_matvec(L, x, i)));
    }
}
