
#include <cmath>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/decomposition/svd.hpp>

#include "../catch_include.hpp"

using namespace zipper;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: verify SVD properties for a given decomposition
// ─────────────────────────────────────────────────────────────────────────────

/// Check that U has orthonormal columns: U^T * U ≈ I.
template <typename UType>
void check_orthonormal_columns(const UType &U, index_type p, double tol) {
    for (index_type i = 0; i < p; ++i) {
        for (index_type j = 0; j < p; ++j) {
            double dot = 0.0;
            for (index_type k = 0; k < U.extent(0); ++k) {
                dot += U(k, i) * U(k, j);
            }
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(dot == Catch::Approx(expected).margin(tol));
        }
    }
}

/// Check that Vt has orthonormal rows: Vt * Vt^T ≈ I.
template <typename VtType>
void check_orthonormal_rows(const VtType &Vt, index_type p, double tol) {
    for (index_type i = 0; i < p; ++i) {
        for (index_type j = 0; j < p; ++j) {
            double dot = 0.0;
            for (index_type k = 0; k < Vt.extent(1); ++k) {
                dot += Vt(i, k) * Vt(j, k);
            }
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(dot == Catch::Approx(expected).margin(tol));
        }
    }
}

/// Check reconstruction: U * diag(S) * Vt ≈ A.
template <typename UType, typename SType, typename VtType, typename AType>
void check_reconstruction(const UType &U,
                          const SType &S,
                          const VtType &Vt,
                          const AType &A,
                          index_type m,
                          index_type n,
                          index_type p,
                          double tol) {
    for (index_type i = 0; i < m; ++i) {
        for (index_type j = 0; j < n; ++j) {
            double sum = 0.0;
            for (index_type k = 0; k < p; ++k) {
                sum += U(i, k) * S(k) * Vt(k, j);
            }
            CHECK(sum == Catch::Approx(A(i, j)).margin(tol));
        }
    }
}

/// Check singular values are non-negative and in descending order.
template <typename SType>
void check_singular_values_sorted(const SType &S, index_type p) {
    for (index_type i = 0; i < p; ++i) { CHECK(S(i) >= 0.0); }
    for (index_type i = 0; i + 1 < p; ++i) { CHECK(S(i) >= S(i + 1) - 1e-12); }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2x2 SVD (exercises fast path)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("svd 2x2 identity", "[decomposition][svd]") {
    Matrix<double, 2, 2> A{{1.0, 0.0}, {0.0, 1.0}};

    auto [U, S, Vt] = utils::decomposition::svd(A);

    CHECK(S(0) == Catch::Approx(1.0).margin(1e-12));
    CHECK(S(1) == Catch::Approx(1.0).margin(1e-12));

    check_orthonormal_columns(U, 2, 1e-12);
    check_orthonormal_rows(Vt, 2, 1e-12);
    check_reconstruction(U, S, Vt, A, 2, 2, 2, 1e-12);
}

TEST_CASE("svd 2x2 diagonal", "[decomposition][svd]") {
    Matrix<double, 2, 2> A{{3.0, 0.0}, {0.0, 5.0}};

    auto [U, S, Vt] = utils::decomposition::svd(A);

    CHECK(S(0) == Catch::Approx(5.0).margin(1e-12));
    CHECK(S(1) == Catch::Approx(3.0).margin(1e-12));

    check_singular_values_sorted(S, 2);
    check_reconstruction(U, S, Vt, A, 2, 2, 2, 1e-12);
}

TEST_CASE("svd 2x2 general", "[decomposition][svd]") {
    Matrix<double, 2, 2> A{{1.0, 2.0}, {3.0, 4.0}};

    auto [U, S, Vt] = utils::decomposition::svd(A);

    check_singular_values_sorted(S, 2);
    check_orthonormal_columns(U, 2, 1e-10);
    check_orthonormal_rows(Vt, 2, 1e-10);
    check_reconstruction(U, S, Vt, A, 2, 2, 2, 1e-10);

    // Known singular values for [[1,2],[3,4]]:
    // sqrt((5 + sqrt(221))/2) ≈ 5.4650, sqrt((5 + sqrt(221))/2 - sqrt(221)) ≈
    // 0.3660 Actually: sigma = sqrt(eigenvalues of A^T A) A^T A = [[10, 14],
    // [14, 20]] eigenvalues: 15 ± sqrt(221) ≈ 15 ± 14.8661 sigma_1 =
    // sqrt(29.8661) ≈ 5.4650 sigma_2 = sqrt(0.1339) ≈ 0.3660
    CHECK(S(0) == Catch::Approx(5.4649857).margin(1e-4));
    CHECK(S(1) == Catch::Approx(0.3659662).margin(1e-4));
}

TEST_CASE("svd 2x2 rank 1", "[decomposition][svd]") {
    Matrix<double, 2, 2> A{{1.0, 2.0}, {2.0, 4.0}};

    auto [U, S, Vt] = utils::decomposition::svd(A);

    check_singular_values_sorted(S, 2);
    CHECK(S(1) == Catch::Approx(0.0).margin(1e-10));
    check_reconstruction(U, S, Vt, A, 2, 2, 2, 1e-10);
}

TEST_CASE("svd 2x2 zero matrix", "[decomposition][svd]") {
    Matrix<double, 2, 2> A{{0.0, 0.0}, {0.0, 0.0}};

    auto [U, S, Vt] = utils::decomposition::svd(A);

    CHECK(S(0) == Catch::Approx(0.0).margin(1e-12));
    CHECK(S(1) == Catch::Approx(0.0).margin(1e-12));
}

// ─────────────────────────────────────────────────────────────────────────────
// 3x3 SVD
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("svd 3x3 identity", "[decomposition][svd]") {
    Matrix<double, 3, 3> A{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    };

    auto [U, S, Vt] = utils::decomposition::svd(A);

    for (index_type i = 0; i < 3; ++i) {
        CHECK(S(i) == Catch::Approx(1.0).margin(1e-12));
    }

    check_orthonormal_columns(U, 3, 1e-12);
    check_orthonormal_rows(Vt, 3, 1e-12);
    check_reconstruction(U, S, Vt, A, 3, 3, 3, 1e-12);
}

TEST_CASE("svd 3x3 diagonal", "[decomposition][svd]") {
    Matrix<double, 3, 3> A{
        {5.0, 0.0, 0.0},
        {0.0, 3.0, 0.0},
        {0.0, 0.0, 1.0},
    };

    auto [U, S, Vt] = utils::decomposition::svd(A);

    CHECK(S(0) == Catch::Approx(5.0).margin(1e-12));
    CHECK(S(1) == Catch::Approx(3.0).margin(1e-12));
    CHECK(S(2) == Catch::Approx(1.0).margin(1e-12));

    check_singular_values_sorted(S, 3);
    check_reconstruction(U, S, Vt, A, 3, 3, 3, 1e-12);
}

TEST_CASE("svd 3x3 general", "[decomposition][svd]") {
    Matrix<double, 3, 3> A{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 10.0}, // Changed from 9 to make full rank.
    };

    auto [U, S, Vt] = utils::decomposition::svd(A);

    check_singular_values_sorted(S, 3);
    check_orthonormal_columns(U, 3, 1e-10);
    check_orthonormal_rows(Vt, 3, 1e-10);
    check_reconstruction(U, S, Vt, A, 3, 3, 3, 1e-10);
}

TEST_CASE("svd 3x3 symmetric positive definite", "[decomposition][svd]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0},
        {1.0, 4.0, 1.0},
        {1.0, 1.0, 4.0},
    };

    auto [U, S, Vt] = utils::decomposition::svd(A);

    // Eigenvalues of this SPD matrix are 6, 3, 3 — these are also the singular
    // values.
    check_singular_values_sorted(S, 3);
    CHECK(S(0) == Catch::Approx(6.0).margin(1e-10));
    CHECK(S(1) == Catch::Approx(3.0).margin(1e-10));
    CHECK(S(2) == Catch::Approx(3.0).margin(1e-10));

    check_reconstruction(U, S, Vt, A, 3, 3, 3, 1e-10);
}

TEST_CASE("svd 3x3 rank deficient", "[decomposition][svd]") {
    // Third column = first + second, so rank 2.
    Matrix<double, 3, 3> A{
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0},
        {1.0, 1.0, 2.0},
    };

    auto [U, S, Vt] = utils::decomposition::svd(A);

    check_singular_values_sorted(S, 3);
    CHECK(S(2) == Catch::Approx(0.0).margin(1e-10));
    check_reconstruction(U, S, Vt, A, 3, 3, 3, 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// Rectangular SVD (tall)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("svd 4x3 tall", "[decomposition][svd]") {
    Matrix<double, 4, 3> A{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0},
        {10.0, 11.0, 13.0},
    };

    auto [U, S, Vt] = utils::decomposition::svd(A);

    // p = min(4, 3) = 3
    CHECK(U.extent(0) == 4);
    CHECK(U.extent(1) == 3);
    CHECK(S.extent(0) == 3);
    CHECK(Vt.extent(0) == 3);
    CHECK(Vt.extent(1) == 3);

    check_singular_values_sorted(S, 3);
    check_orthonormal_columns(U, 3, 1e-10);
    check_orthonormal_rows(Vt, 3, 1e-10);
    check_reconstruction(U, S, Vt, A, 4, 3, 3, 1e-10);
}

TEST_CASE("svd 5x2 tall", "[decomposition][svd]") {
    Matrix<double, 5, 2> A{
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0},
        {7.0, 8.0},
        {9.0, 10.0},
    };

    auto [U, S, Vt] = utils::decomposition::svd(A);

    CHECK(U.extent(0) == 5);
    CHECK(U.extent(1) == 2);
    CHECK(S.extent(0) == 2);
    CHECK(Vt.extent(0) == 2);
    CHECK(Vt.extent(1) == 2);

    check_singular_values_sorted(S, 2);
    check_orthonormal_columns(U, 2, 1e-10);
    check_orthonormal_rows(Vt, 2, 1e-10);
    check_reconstruction(U, S, Vt, A, 5, 2, 2, 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// Rectangular SVD (wide)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("svd 2x4 wide", "[decomposition][svd]") {
    Matrix<double, 2, 4> A{
        {1.0, 2.0, 3.0, 4.0},
        {5.0, 6.0, 7.0, 8.0},
    };

    auto [U, S, Vt] = utils::decomposition::svd(A);

    // p = min(2, 4) = 2
    CHECK(U.extent(0) == 2);
    CHECK(U.extent(1) == 2);
    CHECK(S.extent(0) == 2);
    CHECK(Vt.extent(0) == 2);
    CHECK(Vt.extent(1) == 4);

    check_singular_values_sorted(S, 2);
    check_orthonormal_columns(U, 2, 1e-10);
    check_orthonormal_rows(Vt, 2, 1e-10);
    check_reconstruction(U, S, Vt, A, 2, 4, 2, 1e-10);
}

TEST_CASE("svd 3x5 wide", "[decomposition][svd]") {
    Matrix<double, 3, 5> A{
        {1.0, 0.0, 2.0, 1.0, 3.0},
        {0.0, 1.0, 1.0, 3.0, 2.0},
        {2.0, 1.0, 0.0, 1.0, 4.0},
    };

    auto [U, S, Vt] = utils::decomposition::svd(A);

    CHECK(U.extent(0) == 3);
    CHECK(U.extent(1) == 3);
    CHECK(S.extent(0) == 3);
    CHECK(Vt.extent(0) == 3);
    CHECK(Vt.extent(1) == 5);

    check_singular_values_sorted(S, 3);
    check_orthonormal_columns(U, 3, 1e-10);
    check_orthonormal_rows(Vt, 3, 1e-10);
    check_reconstruction(U, S, Vt, A, 3, 5, 3, 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// 1x1 SVD
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("svd 1x1", "[decomposition][svd]") {
    Matrix<double, 1, 1> A{{-7.0}};

    auto [U, S, Vt] = utils::decomposition::svd(A);

    CHECK(S(0) == Catch::Approx(7.0).margin(1e-12));
    check_reconstruction(U, S, Vt, A, 1, 1, 1, 1e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// Dynamic extent SVD
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("svd dynamic 3x3", "[decomposition][svd]") {
    Matrix<double, std::dynamic_extent, std::dynamic_extent> A(3, 3);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(0, 2) = 3.0;
    A(1, 0) = 4.0;
    A(1, 1) = 5.0;
    A(1, 2) = 6.0;
    A(2, 0) = 7.0;
    A(2, 1) = 8.0;
    A(2, 2) = 10.0;

    auto [U, S, Vt] = utils::decomposition::svd(A);

    check_singular_values_sorted(S, 3);
    check_orthonormal_columns(U, 3, 1e-10);
    check_orthonormal_rows(Vt, 3, 1e-10);
    check_reconstruction(U, S, Vt, A, 3, 3, 3, 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotation matrix SVD (singular values should all be 1)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("svd 3x3 rotation matrix", "[decomposition][svd]") {
    // 90-degree rotation around z-axis.
    Matrix<double, 3, 3> A{
        {0.0, -1.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0},
    };

    auto [U, S, Vt] = utils::decomposition::svd(A);

    for (index_type i = 0; i < 3; ++i) {
        CHECK(S(i) == Catch::Approx(1.0).margin(1e-10));
    }

    check_reconstruction(U, S, Vt, A, 3, 3, 3, 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// Scaled rotation (tests that singular values capture the scaling)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("svd 2x2 scaled rotation", "[decomposition][svd]") {
    // A = s * R where s = 3 and R is a 45-degree rotation.
    double s = 3.0;
    double angle = M_PI / 4.0;
    double c = std::cos(angle);
    double sn = std::sin(angle);
    Matrix<double, 2, 2> A{
        {s * c, -s * sn},
        {s * sn, s * c},
    };

    auto [U, S, Vt] = utils::decomposition::svd(A);

    CHECK(S(0) == Catch::Approx(3.0).margin(1e-10));
    CHECK(S(1) == Catch::Approx(3.0).margin(1e-10));

    check_reconstruction(U, S, Vt, A, 2, 2, 2, 1e-10);
}
