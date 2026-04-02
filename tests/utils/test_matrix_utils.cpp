
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/condition_number.hpp>
#include <zipper/utils/matrix_norm.hpp>
#include <zipper/utils/pseudoinverse.hpp>
#include <zipper/utils/decomposition/schur.hpp>

#include "../catch_include.hpp"

using namespace zipper;

// ═══════════════════════════════════════════════════════════════════════════
// Matrix norms
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("frobenius_norm_identity", "[matrix_norm][frobenius]") {
    Matrix<double, 3, 3> I{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    CHECK(utils::frobenius_norm(I) == Catch::Approx(std::sqrt(3.0)).epsilon(1e-12));
}

TEST_CASE("frobenius_norm_known", "[matrix_norm][frobenius]") {
    // [[1, 2], [3, 4]]: frobenius = sqrt(1+4+9+16) = sqrt(30)
    Matrix<double, 2, 2> M{{{1.0, 2.0}, {3.0, 4.0}}};
    CHECK(utils::frobenius_norm(M) == Catch::Approx(std::sqrt(30.0)).epsilon(1e-12));
}

TEST_CASE("one_norm_identity", "[matrix_norm][one_norm]") {
    Matrix<double, 3, 3> I{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    CHECK(utils::one_norm(I) == Catch::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("one_norm_known", "[matrix_norm][one_norm]") {
    // [[1, -2], [3, 4]]: col0 sum = |1|+|3| = 4, col1 sum = |-2|+|4| = 6
    Matrix<double, 2, 2> M{{{1.0, -2.0}, {3.0, 4.0}}};
    CHECK(utils::one_norm(M) == Catch::Approx(6.0).epsilon(1e-12));
}

TEST_CASE("inf_norm_identity", "[matrix_norm][inf_norm]") {
    Matrix<double, 3, 3> I{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    CHECK(utils::inf_norm(I) == Catch::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("inf_norm_known", "[matrix_norm][inf_norm]") {
    // [[1, -2], [3, 4]]: row0 sum = |1|+|-2| = 3, row1 sum = |3|+|4| = 7
    Matrix<double, 2, 2> M{{{1.0, -2.0}, {3.0, 4.0}}};
    CHECK(utils::inf_norm(M) == Catch::Approx(7.0).epsilon(1e-12));
}

TEST_CASE("spectral_norm_identity", "[matrix_norm][spectral]") {
    Matrix<double, 3, 3> I{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    CHECK(utils::spectral_norm(I) == Catch::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("spectral_norm_diagonal", "[matrix_norm][spectral]") {
    // diag(2, 5, 3): spectral norm = 5
    Matrix<double, 3, 3> D{{{2, 0, 0}, {0, 5, 0}, {0, 0, 3}}};
    CHECK(utils::spectral_norm(D) == Catch::Approx(5.0).epsilon(1e-6));
}

// ═══════════════════════════════════════════════════════════════════════════
// Condition number
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("condition_number_identity", "[condition_number]") {
    Matrix<double, 3, 3> I{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    CHECK(utils::condition_number(I) == Catch::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("condition_number_diagonal", "[condition_number]") {
    // diag(2, 5, 1): cond = 5/1 = 5
    Matrix<double, 3, 3> D{{{2, 0, 0}, {0, 5, 0}, {0, 0, 1}}};
    CHECK(utils::condition_number(D) == Catch::Approx(5.0).epsilon(1e-6));
}

TEST_CASE("condition_number_singular", "[condition_number]") {
    // Singular matrix: one zero row
    Matrix<double, 2, 2> M{{{1, 0}, {0, 0}}};
    CHECK(utils::condition_number(M) == std::numeric_limits<double>::infinity());
}

TEST_CASE("condition_number_2x2", "[condition_number]") {
    // [[2, 1], [1, 2]]: eigenvalues 1 and 3, singular values 1 and 3
    Matrix<double, 2, 2> M{{{2, 1}, {1, 2}}};
    CHECK(utils::condition_number(M) == Catch::Approx(3.0).epsilon(1e-6));
}

// ═══════════════════════════════════════════════════════════════════════════
// Pseudoinverse
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("pseudoinverse_invertible", "[pseudoinverse]") {
    // For invertible matrix, pseudoinverse == inverse
    Matrix<double, 2, 2> M{{{1, 2}, {3, 4}}};
    auto M_pinv = utils::pseudoinverse(M);

    // M * M+ should be close to identity
    auto product = Matrix<double, 2, 2>(M * M_pinv);
    CHECK(product(0, 0) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(product(0, 1) == Catch::Approx(0.0).margin(1e-8));
    CHECK(product(1, 0) == Catch::Approx(0.0).margin(1e-8));
    CHECK(product(1, 1) == Catch::Approx(1.0).epsilon(1e-8));
}

TEST_CASE("pseudoinverse_moore_penrose_conditions", "[pseudoinverse]") {
    // Rectangular matrix: 3x2
    Matrix<double, 3, 2> A{{{1, 0}, {0, 1}, {0, 0}}};
    auto A_pinv = utils::pseudoinverse(A);

    // Condition 1: A * A+ * A = A
    auto AAA = Matrix<double, 3, 2>(
        Matrix<double, 3, 3>(A * A_pinv) * A);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 2; ++j) {
            CHECK(AAA(i, j) == Catch::Approx(A(i, j)).margin(1e-10));
        }
    }

    // Condition 2: A+ * A * A+ = A+
    auto AAp = Matrix<double, 2, 2>(
        Matrix<double, 2, 3>(A_pinv) * Matrix<double, 3, 2>(A));
    auto AApAp = Matrix<double, 2, 3>(AAp * A_pinv);
    for (index_type i = 0; i < 2; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(AApAp(i, j) == Catch::Approx(A_pinv(i, j)).margin(1e-10));
        }
    }
}

TEST_CASE("pseudoinverse_rank_deficient", "[pseudoinverse]") {
    // Rank-1 matrix: [[1, 2], [2, 4]]
    Matrix<double, 2, 2> M{{{1, 2}, {2, 4}}};
    auto M_pinv = utils::pseudoinverse(M);

    // Verify MP condition 1: M * M+ * M = M
    auto MpM = Matrix<double, 2, 2>(M_pinv * M);
    auto MpMM = Matrix<double, 2, 2>(MpM * M_pinv);
    // Actually check A * A+ * A = A
    auto MMp = Matrix<double, 2, 2>(M * M_pinv);
    auto MMpM = Matrix<double, 2, 2>(MMp * M);
    for (index_type i = 0; i < 2; ++i) {
        for (index_type j = 0; j < 2; ++j) {
            CHECK(MMpM(i, j) == Catch::Approx(M(i, j)).margin(1e-8));
        }
    }
}

TEST_CASE("pseudoinverse_identity", "[pseudoinverse]") {
    Matrix<double, 3, 3> I{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    auto I_pinv = utils::pseudoinverse(I);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(I_pinv(i, j) == Catch::Approx(expected).margin(1e-10));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Schur decomposition
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("schur_diagonal", "[schur]") {
    Matrix<double, 3, 3> D{{{2, 0, 0}, {0, 5, 0}, {0, 0, 3}}};
    auto result = utils::decomposition::schur(D);
    REQUIRE(result.has_value());

    auto &U = result->U;
    auto &T = result->T_mat;

    // Verify reconstruction: U * T * U^T ≈ D
    auto recon = Matrix<double, 3, 3>(
        Matrix<double, 3, 3>(U * T) * U.transpose());
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(recon(i, j) == Catch::Approx(D(i, j)).margin(1e-8));
        }
    }

    // T should be (quasi-)upper triangular: T(i,j) ≈ 0 for i > j+1
    for (index_type i = 2; i < 3; ++i) {
        for (index_type j = 0; j < i - 1; ++j) {
            CHECK(T(i, j) == Catch::Approx(0.0).margin(1e-10));
        }
    }
}

TEST_CASE("schur_symmetric", "[schur]") {
    // Symmetric matrix: [[2, 1], [1, 3]]
    Matrix<double, 2, 2> A{{{2, 1}, {1, 3}}};
    auto result = utils::decomposition::schur(A);
    REQUIRE(result.has_value());

    auto &U = result->U;
    auto &T = result->T_mat;

    // Verify reconstruction.
    auto recon = Matrix<double, 2, 2>(
        Matrix<double, 2, 2>(U * T) * U.transpose());
    for (index_type i = 0; i < 2; ++i) {
        for (index_type j = 0; j < 2; ++j) {
            CHECK(recon(i, j) == Catch::Approx(A(i, j)).margin(1e-8));
        }
    }

    // U should be orthogonal: U^T U ≈ I
    auto UtU = Matrix<double, 2, 2>(U.transpose() * U);
    for (index_type i = 0; i < 2; ++i) {
        for (index_type j = 0; j < 2; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(UtU(i, j) == Catch::Approx(expected).margin(1e-8));
        }
    }
}

TEST_CASE("schur_non_symmetric", "[schur]") {
    // Non-symmetric matrix
    Matrix<double, 3, 3> A{{{0, 1, 0}, {0, 0, 1}, {1, 0, 0}}};
    auto result = utils::decomposition::schur(A);
    REQUIRE(result.has_value());

    auto &U = result->U;
    auto &T = result->T_mat;

    // Verify reconstruction.
    auto recon = Matrix<double, 3, 3>(
        Matrix<double, 3, 3>(U * T) * U.transpose());
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(recon(i, j) == Catch::Approx(A(i, j)).margin(1e-8));
        }
    }

    // U should be orthogonal.
    auto UtU = Matrix<double, 3, 3>(U.transpose() * U);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(UtU(i, j) == Catch::Approx(expected).margin(1e-8));
        }
    }

    // T should be quasi-upper triangular (entries below first sub-diagonal are zero).
    for (index_type i = 2; i < 3; ++i) {
        for (index_type j = 0; j < i - 1; ++j) {
            CHECK(T(i, j) == Catch::Approx(0.0).margin(1e-8));
        }
    }
}

TEST_CASE("schur_1x1", "[schur]") {
    Matrix<double, 1, 1> A{{{7.0}}};
    auto result = utils::decomposition::schur(A);
    REQUIRE(result.has_value());
    CHECK(result->T_mat(0, 0) == Catch::Approx(7.0).epsilon(1e-12));
    CHECK(result->U(0, 0) == Catch::Approx(1.0).epsilon(1e-12));
}
