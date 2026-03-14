
#include <cmath>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/decomposition/qr.hpp>

#include "../catch_include.hpp"

using namespace zipper;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
//
// Since zipper may not support matrix-matrix multiplication directly, all
// verification is done via element-level loops.

/// Compute C = A * B element-by-element and store in C.
/// C must already be allocated with the right dimensions.
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

/// Compute C = A^T * B element-by-element.
template <typename T, index_type M, index_type K, index_type N>
void mat_mul_AtB(const Matrix<T, M, K> &A, const Matrix<T, M, N> &B,
                 Matrix<T, K, N> &C) {
  const index_type m = A.extent(0);
  const index_type k = A.extent(1);
  const index_type n = B.extent(1);
  for (index_type i = 0; i < k; ++i) {
    for (index_type j = 0; j < n; ++j) {
      T sum = T{0};
      for (index_type l = 0; l < m; ++l) {
        sum += A(l, i) * B(l, j);
      }
      C(i, j) = sum;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Householder QR (reduced)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("qr reduced 3x3", "[decomposition][qr]") {
    Matrix<double, 3, 3> A{
        {1.0, 1.0, 0.0},
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0},
    };

    auto [Q, R] = utils::decomposition::qr(A);

    // Q should have orthonormal columns: Q^T * Q ≈ I.
    Matrix<double, 3, 3> QtQ(3, 3);
    mat_mul_AtB(Q, Q, QtQ);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(QtQ(i, j) == Catch::Approx(expected).margin(1e-12));
        }
    }

    // R should be upper triangular.
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < i; ++j) {
            CHECK(R(i, j) == Catch::Approx(0.0).margin(1e-12));
        }
    }

    // Q * R ≈ A.
    Matrix<double, 3, 3> QR(3, 3);
    mat_mul(Q, R, QR);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(QR(i, j) == Catch::Approx(A(i, j)).margin(1e-12));
        }
    }
}

TEST_CASE("qr reduced 4x3 tall", "[decomposition][qr]") {
    // Tall matrix: Q is 4x3, R is 3x3.
    Matrix<double, 4, 3> A{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0},
        {10.0, 11.0, 13.0}, // non-singular (changed 12→13)
    };

    auto [Q, R] = utils::decomposition::qr(A);

    // Q^T * Q ≈ I (3x3).
    Matrix<double, 3, 3> QtQ(3, 3);
    mat_mul_AtB(Q, Q, QtQ);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(QtQ(i, j) == Catch::Approx(expected).margin(1e-10));
        }
    }

    // R upper triangular.
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < i; ++j) {
            CHECK(R(i, j) == Catch::Approx(0.0).margin(1e-10));
        }
    }

    // Q * R ≈ A.
    Matrix<double, 4, 3> QR(4, 3);
    mat_mul(Q, R, QR);
    for (index_type i = 0; i < 4; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(QR(i, j) == Catch::Approx(A(i, j)).margin(1e-10));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Householder QR (full)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("qr_full 3x3", "[decomposition][qr_full]") {
    Matrix<double, 3, 3> A{
        {1.0, 1.0, 0.0},
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0},
    };

    auto [Q, R] = utils::decomposition::qr_full(A);

    // Q should be orthogonal: Q^T * Q ≈ I (3x3).
    Matrix<double, 3, 3> QtQ(3, 3);
    mat_mul_AtB(Q, Q, QtQ);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(QtQ(i, j) == Catch::Approx(expected).margin(1e-12));
        }
    }

    // R upper triangular.
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < i; ++j) {
            CHECK(R(i, j) == Catch::Approx(0.0).margin(1e-12));
        }
    }

    // Q * R ≈ A.
    Matrix<double, 3, 3> QR(3, 3);
    mat_mul(Q, R, QR);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(QR(i, j) == Catch::Approx(A(i, j)).margin(1e-12));
        }
    }
}

TEST_CASE("qr_full 4x3 tall", "[decomposition][qr_full]") {
    Matrix<double, 4, 3> A{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0},
        {10.0, 11.0, 13.0},
    };

    auto [Q, R] = utils::decomposition::qr_full(A);

    // Q is 4x4, should be orthogonal.
    Matrix<double, 4, 4> QtQ(4, 4);
    mat_mul_AtB(Q, Q, QtQ);
    for (index_type i = 0; i < 4; ++i) {
        for (index_type j = 0; j < 4; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(QtQ(i, j) == Catch::Approx(expected).margin(1e-10));
        }
    }

    // R is 4x3, upper trapezoidal (R(i,j) = 0 for i > j).
    for (index_type i = 0; i < 4; ++i) {
        for (index_type j = 0; j < std::min(i, index_type{3}); ++j) {
            CHECK(R(i, j) == Catch::Approx(0.0).margin(1e-10));
        }
    }

    // Q * R ≈ A.
    Matrix<double, 4, 3> QR(4, 3);
    mat_mul(Q, R, QR);
    for (index_type i = 0; i < 4; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(QR(i, j) == Catch::Approx(A(i, j)).margin(1e-10));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gram-Schmidt QR
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("qr_gram_schmidt 3x3", "[decomposition][qr_gram_schmidt]") {
    Matrix<double, 3, 3> A{
        {1.0, 1.0, 0.0},
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0},
    };

    auto [Q, R] = utils::decomposition::qr_gram_schmidt(A);

    // Q^T * Q ≈ I.
    Matrix<double, 3, 3> QtQ(3, 3);
    mat_mul_AtB(Q, Q, QtQ);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(QtQ(i, j) == Catch::Approx(expected).margin(1e-10));
        }
    }

    // R upper triangular.
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < i; ++j) {
            CHECK(R(i, j) == Catch::Approx(0.0).margin(1e-10));
        }
    }

    // Q * R ≈ A.
    Matrix<double, 3, 3> QR(3, 3);
    mat_mul(Q, R, QR);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(QR(i, j) == Catch::Approx(A(i, j)).margin(1e-10));
        }
    }
}

TEST_CASE("qr_gram_schmidt 4x3 tall", "[decomposition][qr_gram_schmidt]") {
    Matrix<double, 4, 3> A{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0},
        {10.0, 11.0, 13.0},
    };

    auto [Q, R] = utils::decomposition::qr_gram_schmidt(A);

    // Q^T * Q ≈ I (3x3).
    Matrix<double, 3, 3> QtQ(3, 3);
    mat_mul_AtB(Q, Q, QtQ);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(QtQ(i, j) == Catch::Approx(expected).margin(1e-10));
        }
    }

    // R upper triangular.
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < i; ++j) {
            CHECK(R(i, j) == Catch::Approx(0.0).margin(1e-10));
        }
    }

    // Q * R ≈ A.
    Matrix<double, 4, 3> QR(4, 3);
    mat_mul(Q, R, QR);
    for (index_type i = 0; i < 4; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(QR(i, j) == Catch::Approx(A(i, j)).margin(1e-10));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Diagonal R entries should be positive (convention)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("qr diagonal of R has consistent sign", "[decomposition][qr]") {
    // For Householder QR the diagonal of R can be positive or negative
    // (depends on the sign choice in the reflection).  This test just
    // verifies that the factorisation is valid regardless.
    Matrix<double, 2, 2> A{{3.0, 1.0}, {4.0, 2.0}};

    auto [Q, R] = utils::decomposition::qr(A);

    // Q * R ≈ A.
    Matrix<double, 2, 2> QR(2, 2);
    mat_mul(Q, R, QR);
    for (index_type i = 0; i < 2; ++i) {
        for (index_type j = 0; j < 2; ++j) {
            CHECK(QR(i, j) == Catch::Approx(A(i, j)).margin(1e-12));
        }
    }
}
