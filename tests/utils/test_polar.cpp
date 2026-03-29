
#include <cmath>

#include <zipper/Matrix.hpp>
#include <zipper/utils/decomposition/polar.hpp>
#include <zipper/utils/determinant.hpp>

#include "../catch_include.hpp"

using namespace zipper;
using namespace zipper::utils::decomposition;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Verify R is a proper rotation: R^T R = I, det R = +1.
template <typename MatType>
void check_proper_rotation(const MatType &R, index_type n, double tol) {
    // R^T * R = I
    for (index_type i = 0; i < n; ++i) {
        for (index_type j = 0; j < n; ++j) {
            double dot = 0.0;
            for (index_type k = 0; k < n; ++k) { dot += R(k, i) * R(k, j); }
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(dot == Catch::Approx(expected).margin(tol));
        }
    }
    // det R = +1
    double det = utils::determinant(R);
    CHECK(det == Catch::Approx(1.0).margin(tol));
}

/// Verify S is symmetric.
template <typename MatType>
void check_symmetric(const MatType &S, index_type n, double tol) {
    for (index_type i = 0; i < n; ++i) {
        for (index_type j = i + 1; j < n; ++j) {
            CHECK(S(i, j) == Catch::Approx(S(j, i)).margin(tol));
        }
    }
}

/// Verify R * S = F.
template <typename RType, typename SType, typename FType>
void check_reconstruction(const RType &R,
                          const SType &S,
                          const FType &F,
                          index_type n,
                          double tol) {
    for (index_type i = 0; i < n; ++i) {
        for (index_type j = 0; j < n; ++j) {
            double rs = 0.0;
            for (index_type k = 0; k < n; ++k) { rs += R(i, k) * S(k, j); }
            CHECK(rs
                  == Catch::Approx(static_cast<double>(F(i, j))).margin(tol));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2x2 tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("polar_2x2_identity", "[decomposition][polar]") {
    Matrix<double, 2, 2> F;
    F(0, 0) = 1.0;
    F(0, 1) = 0.0;
    F(1, 0) = 0.0;
    F(1, 1) = 1.0;

    auto [R, S] = polar(F);

    check_proper_rotation(R, 2, 1e-10);
    check_symmetric(S, 2, 1e-10);
    check_reconstruction(R, S, F, 2, 1e-10);

    // R = I, S = I
    for (index_type i = 0; i < 2; ++i) {
        for (index_type j = 0; j < 2; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(R(i, j) == Catch::Approx(expected).margin(1e-10));
            CHECK(S(i, j) == Catch::Approx(expected).margin(1e-10));
        }
    }
}

TEST_CASE("polar_2x2_pure_rotation", "[decomposition][polar]") {
    constexpr double theta = 0.7;
    Matrix<double, 2, 2> F;
    F(0, 0) = std::cos(theta);
    F(0, 1) = -std::sin(theta);
    F(1, 0) = std::sin(theta);
    F(1, 1) = std::cos(theta);

    auto [R, S] = polar(F);

    check_proper_rotation(R, 2, 1e-10);
    check_symmetric(S, 2, 1e-10);
    check_reconstruction(R, S, F, 2, 1e-10);

    // R should equal F
    for (index_type i = 0; i < 2; ++i) {
        for (index_type j = 0; j < 2; ++j) {
            CHECK(R(i, j) == Catch::Approx(F(i, j)).margin(1e-10));
        }
    }

    // S should be identity
    CHECK(S(0, 0) == Catch::Approx(1.0).margin(1e-10));
    CHECK(S(0, 1) == Catch::Approx(0.0).margin(1e-10));
    CHECK(S(1, 0) == Catch::Approx(0.0).margin(1e-10));
    CHECK(S(1, 1) == Catch::Approx(1.0).margin(1e-10));
}

TEST_CASE("polar_2x2_scaled_rotation", "[decomposition][polar]") {
    constexpr double theta = 1.2;
    constexpr double s = 2.5;

    Matrix<double, 2, 2> F;
    F(0, 0) = s * std::cos(theta);
    F(0, 1) = -s * std::sin(theta);
    F(1, 0) = s * std::sin(theta);
    F(1, 1) = s * std::cos(theta);

    auto [R, S] = polar(F);

    check_proper_rotation(R, 2, 1e-10);
    check_symmetric(S, 2, 1e-10);
    check_reconstruction(R, S, F, 2, 1e-10);

    // R should be the rotation
    CHECK(R(0, 0) == Catch::Approx(std::cos(theta)).margin(1e-10));
    CHECK(R(0, 1) == Catch::Approx(-std::sin(theta)).margin(1e-10));
    CHECK(R(1, 0) == Catch::Approx(std::sin(theta)).margin(1e-10));
    CHECK(R(1, 1) == Catch::Approx(std::cos(theta)).margin(1e-10));

    // S should be s*I
    CHECK(S(0, 0) == Catch::Approx(s).margin(1e-10));
    CHECK(S(0, 1) == Catch::Approx(0.0).margin(1e-10));
    CHECK(S(1, 0) == Catch::Approx(0.0).margin(1e-10));
    CHECK(S(1, 1) == Catch::Approx(s).margin(1e-10));
}

TEST_CASE("polar_2x2_anisotropic_stretch", "[decomposition][polar]") {
    Matrix<double, 2, 2> F;
    F(0, 0) = 2.0;
    F(0, 1) = 0.0;
    F(1, 0) = 0.0;
    F(1, 1) = 3.0;

    auto [R, S] = polar(F);

    check_proper_rotation(R, 2, 1e-10);
    check_symmetric(S, 2, 1e-10);
    check_reconstruction(R, S, F, 2, 1e-10);

    // R should be identity
    CHECK(R(0, 0) == Catch::Approx(1.0).margin(1e-10));
    CHECK(R(0, 1) == Catch::Approx(0.0).margin(1e-10));
    CHECK(R(1, 0) == Catch::Approx(0.0).margin(1e-10));
    CHECK(R(1, 1) == Catch::Approx(1.0).margin(1e-10));

    // S should be diag(2, 3)
    CHECK(S(0, 0) == Catch::Approx(2.0).margin(1e-10));
    CHECK(S(0, 1) == Catch::Approx(0.0).margin(1e-10));
    CHECK(S(1, 0) == Catch::Approx(0.0).margin(1e-10));
    CHECK(S(1, 1) == Catch::Approx(3.0).margin(1e-10));
}

TEST_CASE("polar_2x2_general", "[decomposition][polar]") {
    Matrix<double, 2, 2> F;
    F(0, 0) = 1.5;
    F(0, 1) = 0.3;
    F(1, 0) = -0.7;
    F(1, 1) = 2.1;

    auto [R, S] = polar(F);

    check_proper_rotation(R, 2, 1e-10);
    check_symmetric(S, 2, 1e-10);
    check_reconstruction(R, S, F, 2, 1e-10);
}

TEST_CASE("polar_2x2_reflection_correction", "[decomposition][polar]") {
    // F with negative determinant — should still produce proper rotation.
    Matrix<double, 2, 2> F;
    F(0, 0) = -1.0;
    F(0, 1) = 0.0;
    F(1, 0) = 0.0;
    F(1, 1) = 1.0;

    auto [R, S] = polar(F);

    check_proper_rotation(R, 2, 1e-10);
    check_symmetric(S, 2, 1e-10);
    check_reconstruction(R, S, F, 2, 1e-10);

    double det_F = utils::determinant(F);
    CHECK(det_F < 0.0); // F is a reflection

    // R is still proper rotation (det = +1)
    double det_R = utils::determinant(R);
    CHECK(det_R == Catch::Approx(1.0).margin(1e-10));
}

// ─────────────────────────────────────────────────────────────────────────────
// 3x3 tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("polar_3x3_identity", "[decomposition][polar]") {
    Matrix<double, 3, 3> F;
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) { F(i, j) = (i == j) ? 1.0 : 0.0; }
    }

    auto [R, S] = polar(F);

    check_proper_rotation(R, 3, 1e-10);
    check_symmetric(S, 3, 1e-10);
    check_reconstruction(R, S, F, 3, 1e-10);

    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(R(i, j) == Catch::Approx(expected).margin(1e-10));
            CHECK(S(i, j) == Catch::Approx(expected).margin(1e-10));
        }
    }
}

TEST_CASE("polar_3x3_rotation_x", "[decomposition][polar]") {
    // Rotation about X axis
    constexpr double theta = 0.9;
    Matrix<double, 3, 3> F;
    F(0, 0) = 1.0;
    F(0, 1) = 0.0;
    F(0, 2) = 0.0;
    F(1, 0) = 0.0;
    F(1, 1) = std::cos(theta);
    F(1, 2) = -std::sin(theta);
    F(2, 0) = 0.0;
    F(2, 1) = std::sin(theta);
    F(2, 2) = std::cos(theta);

    auto [R, S] = polar(F);

    check_proper_rotation(R, 3, 1e-10);
    check_symmetric(S, 3, 1e-10);
    check_reconstruction(R, S, F, 3, 1e-10);

    // R should equal F (pure rotation)
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(R(i, j) == Catch::Approx(F(i, j)).margin(1e-10));
        }
    }

    // S should be identity
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(S(i, j) == Catch::Approx(expected).margin(1e-10));
        }
    }
}

TEST_CASE("polar_3x3_scaled", "[decomposition][polar]") {
    // F = s * Rx(theta)
    constexpr double theta = 0.5;
    constexpr double s = 3.0;
    Matrix<double, 3, 3> F;
    F(0, 0) = s;
    F(0, 1) = 0.0;
    F(0, 2) = 0.0;
    F(1, 0) = 0.0;
    F(1, 1) = s * std::cos(theta);
    F(1, 2) = -s * std::sin(theta);
    F(2, 0) = 0.0;
    F(2, 1) = s * std::sin(theta);
    F(2, 2) = s * std::cos(theta);

    auto [R, S] = polar(F);

    check_proper_rotation(R, 3, 1e-10);
    check_symmetric(S, 3, 1e-10);
    check_reconstruction(R, S, F, 3, 1e-10);

    // S should be s*I
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? s : 0.0;
            CHECK(S(i, j) == Catch::Approx(expected).margin(1e-10));
        }
    }
}

TEST_CASE("polar_3x3_general", "[decomposition][polar]") {
    Matrix<double, 3, 3> F;
    F(0, 0) = 2.0;
    F(0, 1) = 0.5;
    F(0, 2) = -0.3;
    F(1, 0) = 0.1;
    F(1, 1) = 1.5;
    F(1, 2) = 0.7;
    F(2, 0) = -0.4;
    F(2, 1) = 0.2;
    F(2, 2) = 3.0;

    auto [R, S] = polar(F);

    check_proper_rotation(R, 3, 1e-10);
    check_symmetric(S, 3, 1e-10);
    check_reconstruction(R, S, F, 3, 1e-10);
}

TEST_CASE("polar_3x3_anisotropic_stretch", "[decomposition][polar]") {
    Matrix<double, 3, 3> F;
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) { F(i, j) = 0.0; }
    }
    F(0, 0) = 1.0;
    F(1, 1) = 2.0;
    F(2, 2) = 5.0;

    auto [R, S] = polar(F);

    check_proper_rotation(R, 3, 1e-10);
    check_symmetric(S, 3, 1e-10);
    check_reconstruction(R, S, F, 3, 1e-10);

    // R should be identity
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(R(i, j) == Catch::Approx(expected).margin(1e-10));
        }
    }

    // S should be F itself
    CHECK(S(0, 0) == Catch::Approx(1.0).margin(1e-10));
    CHECK(S(1, 1) == Catch::Approx(2.0).margin(1e-10));
    CHECK(S(2, 2) == Catch::Approx(5.0).margin(1e-10));
}

TEST_CASE("polar_3x3_reflection_correction", "[decomposition][polar]") {
    // F with det < 0 (reflection in x)
    Matrix<double, 3, 3> F;
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) { F(i, j) = (i == j) ? 1.0 : 0.0; }
    }
    F(0, 0) = -1.0;

    auto [R, S] = polar(F);

    check_proper_rotation(R, 3, 1e-10);
    check_symmetric(S, 3, 1e-10);
    check_reconstruction(R, S, F, 3, 1e-10);

    // det(F) < 0
    double det_F = utils::determinant(F);
    CHECK(det_F < 0.0);

    // R is still proper
    double det_R = utils::determinant(R);
    CHECK(det_R == Catch::Approx(1.0).margin(1e-10));
}

// ─────────────────────────────────────────────────────────────────────────────
// Dynamic extent test
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("polar_dynamic_3x3", "[decomposition][polar]") {
    Matrix<double, std::dynamic_extent, std::dynamic_extent> F(3, 3);
    F(0, 0) = 1.2;
    F(0, 1) = 0.3;
    F(0, 2) = -0.1;
    F(1, 0) = -0.4;
    F(1, 1) = 2.0;
    F(1, 2) = 0.5;
    F(2, 0) = 0.2;
    F(2, 1) = -0.1;
    F(2, 2) = 1.8;

    auto [R, S] = polar(F);

    check_proper_rotation(R, 3, 1e-9);
    check_symmetric(S, 3, 1e-9);
    check_reconstruction(R, S, F, 3, 1e-9);
}
