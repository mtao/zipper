
#include <cmath>
#include <complex>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/eigenvalue/general.hpp>

#include "../catch_include.hpp"

using namespace zipper;

// ═══════════════════════════════════════════════════════════════════════════
// Tridiagonal QR
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("tridiag_qr_diagonal", "[eigenvalue][tridiagonal]") {
    // Diagonal matrix: eigenvalues are the diagonal entries.
    Matrix<double, 3, 3> T{{{2, 0, 0}, {0, 5, 0}, {0, 0, 3}}};
    auto result = utils::eigenvalue::tridiagonal_qr_eigen(T);
    REQUIRE(result.has_value());

    // Eigenvalues should be {2, 3, 5} ascending.
    CHECK(result->eigenvalues(0) == Catch::Approx(2.0).epsilon(1e-10));
    CHECK(result->eigenvalues(1) == Catch::Approx(3.0).epsilon(1e-10));
    CHECK(result->eigenvalues(2) == Catch::Approx(5.0).epsilon(1e-10));
}

TEST_CASE("tridiag_qr_2x2", "[eigenvalue][tridiagonal]") {
    // [[2, 1], [1, 3]]: eigenvalues (5 +/- sqrt(5))/2
    Matrix<double, 2, 2> T{{{2, 1}, {1, 3}}};
    auto result = utils::eigenvalue::tridiagonal_qr_eigen(T);
    REQUIRE(result.has_value());

    double e1 = (5.0 - std::sqrt(5.0)) / 2.0;
    double e2 = (5.0 + std::sqrt(5.0)) / 2.0;
    CHECK(result->eigenvalues(0) == Catch::Approx(e1).epsilon(1e-10));
    CHECK(result->eigenvalues(1) == Catch::Approx(e2).epsilon(1e-10));
}

TEST_CASE("tridiag_qr_reconstruction", "[eigenvalue][tridiagonal]") {
    // Tridiagonal: [[3, 1, 0], [1, 2, 1], [0, 1, 3]]
    Matrix<double, 3, 3> T{{{3, 1, 0}, {1, 2, 1}, {0, 1, 3}}};
    auto result = utils::eigenvalue::tridiagonal_qr_eigen(T);
    REQUIRE(result.has_value());

    auto &Q = result->eigenvectors;

    // Verify Q^T Q ≈ I
    auto QtQ = Matrix<double, 3, 3>(Q.transpose() * Q);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(QtQ(i, j) == Catch::Approx(expected).margin(1e-10));
        }
    }

    // Verify T * v_k = lambda_k * v_k for each pair
    for (index_type k = 0; k < 3; ++k) {
        auto Tv = Vector<double, 3>(T * Q.col(k));
        auto lv = Vector<double, 3>(result->eigenvalues(k) * Q.col(k));
        for (index_type i = 0; i < 3; ++i) {
            CHECK(Tv(i) == Catch::Approx(lv(i)).margin(1e-10));
        }
    }
}

TEST_CASE("tridiag_qr_1x1", "[eigenvalue][tridiagonal]") {
    Matrix<double, 1, 1> T{{{7.0}}};
    auto result = utils::eigenvalue::tridiagonal_qr_eigen(T);
    REQUIRE(result.has_value());
    CHECK(result->eigenvalues(0) == Catch::Approx(7.0).epsilon(1e-12));
}

// ═══════════════════════════════════════════════════════════════════════════
// Symmetric eigenvalue
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("symmetric_eigen_2x2", "[eigenvalue][symmetric]") {
    // [[2, 1], [1, 2]]: eigenvalues 1 and 3
    Matrix<double, 2, 2> M{{{2, 1}, {1, 2}}};
    auto result = utils::eigenvalue::symmetric_eigen(M);
    REQUIRE(result.has_value());

    CHECK(result->eigenvalues(0) == Catch::Approx(1.0).epsilon(1e-10));
    CHECK(result->eigenvalues(1) == Catch::Approx(3.0).epsilon(1e-10));
}

TEST_CASE("symmetric_eigen_3x3_known", "[eigenvalue][symmetric]") {
    // [[4, 1, 1], [1, 4, 1], [1, 1, 4]]: eigenvalues 3, 3, 6
    Matrix<double, 3, 3> M{{{4, 1, 1}, {1, 4, 1}, {1, 1, 4}}};
    auto result = utils::eigenvalue::symmetric_eigen(M);
    REQUIRE(result.has_value());

    CHECK(result->eigenvalues(0) == Catch::Approx(3.0).epsilon(1e-8));
    CHECK(result->eigenvalues(1) == Catch::Approx(3.0).epsilon(1e-8));
    CHECK(result->eigenvalues(2) == Catch::Approx(6.0).epsilon(1e-8));
}

TEST_CASE("symmetric_eigen_diagonal", "[eigenvalue][symmetric]") {
    Matrix<double, 3, 3> D{{{2, 0, 0}, {0, 5, 0}, {0, 0, 3}}};
    auto result = utils::eigenvalue::symmetric_eigen(D);
    REQUIRE(result.has_value());

    CHECK(result->eigenvalues(0) == Catch::Approx(2.0).epsilon(1e-10));
    CHECK(result->eigenvalues(1) == Catch::Approx(3.0).epsilon(1e-10));
    CHECK(result->eigenvalues(2) == Catch::Approx(5.0).epsilon(1e-10));
}

TEST_CASE("symmetric_eigen_reconstruction", "[eigenvalue][symmetric]") {
    // Verify M * v_k = lambda_k * v_k
    Matrix<double, 3, 3> M{{{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}}};
    auto result = utils::eigenvalue::symmetric_eigen(M);
    REQUIRE(result.has_value());

    auto &V = result->eigenvectors;

    // Orthogonality: V^T V ≈ I
    auto VtV = Matrix<double, 3, 3>(V.transpose() * V);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(VtV(i, j) == Catch::Approx(expected).margin(1e-10));
        }
    }

    // Eigenpair verification
    for (index_type k = 0; k < 3; ++k) {
        auto Mv = Vector<double, 3>(M * V.col(k));
        auto lv = Vector<double, 3>(result->eigenvalues(k) * V.col(k));
        for (index_type i = 0; i < 3; ++i) {
            CHECK(Mv(i) == Catch::Approx(lv(i)).margin(1e-8));
        }
    }
}

TEST_CASE("symmetric_eigen_negative_eigenvalues", "[eigenvalue][symmetric]") {
    // [[-2, 1], [1, -2]]: eigenvalues -3, -1
    Matrix<double, 2, 2> M{{{-2, 1}, {1, -2}}};
    auto result = utils::eigenvalue::symmetric_eigen(M);
    REQUIRE(result.has_value());

    CHECK(result->eigenvalues(0) == Catch::Approx(-3.0).epsilon(1e-10));
    CHECK(result->eigenvalues(1) == Catch::Approx(-1.0).epsilon(1e-10));
}

TEST_CASE("symmetric_eigen_1x1", "[eigenvalue][symmetric]") {
    Matrix<double, 1, 1> M{{{-5.0}}};
    auto result = utils::eigenvalue::symmetric_eigen(M);
    REQUIRE(result.has_value());
    CHECK(result->eigenvalues(0) == Catch::Approx(-5.0).epsilon(1e-12));
}

TEST_CASE("symmetric_eigen_identity", "[eigenvalue][symmetric]") {
    Matrix<double, 3, 3> I{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    auto result = utils::eigenvalue::symmetric_eigen(I);
    REQUIRE(result.has_value());

    for (index_type i = 0; i < 3; ++i) {
        CHECK(result->eigenvalues(i) == Catch::Approx(1.0).epsilon(1e-10));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// General eigenvalue (Hessenberg QR)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("general_eigen_diagonal", "[eigenvalue][general]") {
    Matrix<double, 3, 3> D{{{2, 0, 0}, {0, 5, 0}, {0, 0, 3}}};
    auto result = utils::eigenvalue::hessenberg_qr_eigen(D);
    REQUIRE(result.has_value());

    // Eigenvalues should be 2, 3, 5 (but order depends on Schur form).
    // Collect real parts and sort.
    std::vector<double> evals;
    for (index_type i = 0; i < 3; ++i) {
        CHECK(result->eigenvalues(i).imag() == Catch::Approx(0.0).margin(1e-10));
        evals.push_back(result->eigenvalues(i).real());
    }
    std::sort(evals.begin(), evals.end());
    CHECK(evals[0] == Catch::Approx(2.0).epsilon(1e-8));
    CHECK(evals[1] == Catch::Approx(3.0).epsilon(1e-8));
    CHECK(evals[2] == Catch::Approx(5.0).epsilon(1e-8));
}

TEST_CASE("general_eigen_2x2_real", "[eigenvalue][general]") {
    // [[2, 1], [1, 2]]: eigenvalues 1, 3 (real)
    Matrix<double, 2, 2> M{{{2, 1}, {1, 2}}};
    auto result = utils::eigenvalue::hessenberg_qr_eigen(M);
    REQUIRE(result.has_value());

    std::vector<double> evals;
    for (index_type i = 0; i < 2; ++i) {
        CHECK(result->eigenvalues(i).imag() == Catch::Approx(0.0).margin(1e-10));
        evals.push_back(result->eigenvalues(i).real());
    }
    std::sort(evals.begin(), evals.end());
    CHECK(evals[0] == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(evals[1] == Catch::Approx(3.0).epsilon(1e-8));
}

TEST_CASE("general_eigen_rotation_complex", "[eigenvalue][general]") {
    // 90-degree rotation matrix: eigenvalues are +/- i
    Matrix<double, 2, 2> M{{{0, -1}, {1, 0}}};
    auto result = utils::eigenvalue::hessenberg_qr_eigen(M);
    REQUIRE(result.has_value());

    // Eigenvalues should be i and -i (|eigenvalue| = 1)
    for (index_type i = 0; i < 2; ++i) {
        CHECK(std::abs(result->eigenvalues(i)) == Catch::Approx(1.0).epsilon(1e-8));
    }

    // Real parts should be ~0
    for (index_type i = 0; i < 2; ++i) {
        CHECK(result->eigenvalues(i).real() == Catch::Approx(0.0).margin(1e-8));
    }

    // Imaginary parts should be +1 and -1 (in some order)
    std::vector<double> imags = {result->eigenvalues(0).imag(),
                                  result->eigenvalues(1).imag()};
    std::sort(imags.begin(), imags.end());
    CHECK(imags[0] == Catch::Approx(-1.0).epsilon(1e-8));
    CHECK(imags[1] == Catch::Approx(1.0).epsilon(1e-8));
}

TEST_CASE("general_eigen_non_symmetric", "[eigenvalue][general]") {
    // Companion matrix for x^3 - 1 = 0: eigenvalues are cube roots of unity
    // [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    Matrix<double, 3, 3> M{{{0, 0, 1}, {1, 0, 0}, {0, 1, 0}}};
    auto result = utils::eigenvalue::hessenberg_qr_eigen(M);
    REQUIRE(result.has_value());

    // All eigenvalues should have |lambda| = 1
    for (index_type i = 0; i < 3; ++i) {
        CHECK(std::abs(result->eigenvalues(i)) == Catch::Approx(1.0).epsilon(1e-6));
    }

    // One eigenvalue should be real (= 1)
    bool found_real = false;
    for (index_type i = 0; i < 3; ++i) {
        if (std::abs(result->eigenvalues(i).imag()) < 1e-6) {
            CHECK(result->eigenvalues(i).real() == Catch::Approx(1.0).epsilon(1e-6));
            found_real = true;
        }
    }
    CHECK(found_real);
}

TEST_CASE("general_eigen_1x1", "[eigenvalue][general]") {
    Matrix<double, 1, 1> M{{{-3.0}}};
    auto result = utils::eigenvalue::hessenberg_qr_eigen(M);
    REQUIRE(result.has_value());
    CHECK(result->eigenvalues(0).real() == Catch::Approx(-3.0).epsilon(1e-12));
    CHECK(result->eigenvalues(0).imag() == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("general_eigen_schur_orthogonality", "[eigenvalue][general]") {
    Matrix<double, 3, 3> M{{{1, 2, 3}, {0, 4, 5}, {0, 0, 6}}};
    auto result = utils::eigenvalue::hessenberg_qr_eigen(M);
    REQUIRE(result.has_value());

    // Schur vectors should be orthogonal
    auto &U = result->schur_vectors;
    auto UtU = Matrix<double, 3, 3>(U.transpose() * U);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(UtU(i, j) == Catch::Approx(expected).margin(1e-8));
        }
    }

    // Eigenvalues of upper triangular = diagonal entries
    std::vector<double> evals;
    for (index_type i = 0; i < 3; ++i) {
        evals.push_back(result->eigenvalues(i).real());
    }
    std::sort(evals.begin(), evals.end());
    CHECK(evals[0] == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(evals[1] == Catch::Approx(4.0).epsilon(1e-8));
    CHECK(evals[2] == Catch::Approx(6.0).epsilon(1e-8));
}
