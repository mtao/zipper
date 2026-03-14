
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/solver/jacobi.hpp>
#include <zipper/utils/solver/gauss_seidel.hpp>
#include <zipper/utils/solver/conjugate_gradient.hpp>
#include <zipper/utils/solver/gmres.hpp>
#include <zipper/utils/solver/bicgstab.hpp>

#include "../catch_include.hpp"

using namespace zipper;

// ─────────────────────────────────────────────────────────────────────────────
// Test systems
// ─────────────────────────────────────────────────────────────────────────────
//
// System 1 (2x2 SPD, diagonally dominant):
//   A = [[4, 1], [1, 3]]
//   b = [1, 2]
//   x = [1/11, 7/11]
//
// System 2 (3x3 SPD, diagonally dominant):
//   A = [[4, 1, 1], [1, 4, 1], [1, 1, 4]]
//   b = [6, 6, 6]
//   x = [1, 1, 1]
//
// System 3 (3x3 upper triangular, non-symmetric -- for GMRES/BiCGSTAB):
//   A = [[3, 1, 0], [0, 3, 1], [0, 0, 3]]
//   b = [4, 4, 3]
//   x = [1, 1, 1]

// ─────────────────────────────────────────────────────────────────────────────
// Jacobi
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("jacobi 2x2 SPD", "[solver][jacobi]") {
    Matrix<double, 2, 2> A{{4.0, 1.0}, {1.0, 3.0}};
    Vector<double, 2> b{1.0, 2.0};

    auto result = utils::solver::jacobi(A, b);
    REQUIRE(result.has_value());

    CHECK(result->x(0) == Catch::Approx(1.0 / 11.0).epsilon(1e-8));
    CHECK(result->x(1) == Catch::Approx(7.0 / 11.0).epsilon(1e-8));
    CHECK(result->residual_norm < 1e-10);
}

TEST_CASE("jacobi 3x3 SPD", "[solver][jacobi]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> b{6.0, 6.0, 6.0};

    auto result = utils::solver::jacobi(A, b);
    REQUIRE(result.has_value());

    CHECK(result->x(0) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(1) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(2) == Catch::Approx(1.0).epsilon(1e-8));
}

TEST_CASE("jacobi with initial guess", "[solver][jacobi]") {
    Matrix<double, 2, 2> A{{4.0, 1.0}, {1.0, 3.0}};
    Vector<double, 2> b{1.0, 2.0};
    Vector<double, 2> x0{0.1, 0.6}; // close to true solution

    auto result = utils::solver::jacobi(A, b, x0);
    REQUIRE(result.has_value());

    CHECK(result->x(0) == Catch::Approx(1.0 / 11.0).epsilon(1e-8));
    CHECK(result->x(1) == Catch::Approx(7.0 / 11.0).epsilon(1e-8));
    // Should converge faster with a good initial guess.
}

// ─────────────────────────────────────────────────────────────────────────────
// Gauss-Seidel
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("gauss_seidel 2x2 SPD", "[solver][gauss_seidel]") {
    Matrix<double, 2, 2> A{{4.0, 1.0}, {1.0, 3.0}};
    Vector<double, 2> b{1.0, 2.0};

    auto result = utils::solver::gauss_seidel(A, b);
    REQUIRE(result.has_value());

    CHECK(result->x(0) == Catch::Approx(1.0 / 11.0).epsilon(1e-8));
    CHECK(result->x(1) == Catch::Approx(7.0 / 11.0).epsilon(1e-8));
}

TEST_CASE("gauss_seidel 3x3 SPD", "[solver][gauss_seidel]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> b{6.0, 6.0, 6.0};

    auto result = utils::solver::gauss_seidel(A, b);
    REQUIRE(result.has_value());

    CHECK(result->x(0) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(1) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(2) == Catch::Approx(1.0).epsilon(1e-8));
}

TEST_CASE("gauss_seidel converges faster than jacobi", "[solver][gauss_seidel]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> b{6.0, 6.0, 6.0};

    auto gs = utils::solver::gauss_seidel(A, b);
    auto jac = utils::solver::jacobi(A, b);
    REQUIRE(gs.has_value());
    REQUIRE(jac.has_value());

    // Gauss-Seidel should need fewer iterations than Jacobi.
    CHECK(gs->iterations <= jac->iterations);
}

// ─────────────────────────────────────────────────────────────────────────────
// Conjugate Gradient
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("cg 2x2 SPD", "[solver][cg]") {
    Matrix<double, 2, 2> A{{4.0, 1.0}, {1.0, 3.0}};
    Vector<double, 2> b{1.0, 2.0};

    auto result = utils::solver::conjugate_gradient(A, b);
    REQUIRE(result.has_value());

    CHECK(result->x(0) == Catch::Approx(1.0 / 11.0).epsilon(1e-8));
    CHECK(result->x(1) == Catch::Approx(7.0 / 11.0).epsilon(1e-8));
}

TEST_CASE("cg 3x3 SPD", "[solver][cg]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> b{6.0, 6.0, 6.0};

    auto result = utils::solver::conjugate_gradient(A, b);
    REQUIRE(result.has_value());

    CHECK(result->x(0) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(1) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(2) == Catch::Approx(1.0).epsilon(1e-8));
    // CG should converge in at most n iterations for an n x n SPD matrix.
    CHECK(result->iterations <= 3);
}

TEST_CASE("cg converges in at most n steps", "[solver][cg]") {
    // For a 2x2 SPD system, CG should converge in at most 2 iterations.
    Matrix<double, 2, 2> A{{4.0, 1.0}, {1.0, 3.0}};
    Vector<double, 2> b{1.0, 2.0};

    auto result = utils::solver::conjugate_gradient(A, b);
    REQUIRE(result.has_value());
    CHECK(result->iterations <= 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// GMRES
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("gmres 2x2 SPD", "[solver][gmres]") {
    Matrix<double, 2, 2> A{{4.0, 1.0}, {1.0, 3.0}};
    Vector<double, 2> b{1.0, 2.0};

    auto result = utils::solver::gmres(A, b);
    REQUIRE(result.has_value());

    CHECK(result->x(0) == Catch::Approx(1.0 / 11.0).epsilon(1e-8));
    CHECK(result->x(1) == Catch::Approx(7.0 / 11.0).epsilon(1e-8));
}

TEST_CASE("gmres 3x3 non-symmetric", "[solver][gmres]") {
    Matrix<double, 3, 3> A{
        {3.0, 1.0, 0.0}, {0.0, 3.0, 1.0}, {0.0, 0.0, 3.0}};
    Vector<double, 3> b{4.0, 4.0, 3.0};

    auto result = utils::solver::gmres(A, b);
    REQUIRE(result.has_value());

    CHECK(result->x(0) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(1) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(2) == Catch::Approx(1.0).epsilon(1e-8));
}

TEST_CASE("gmres 3x3 SPD", "[solver][gmres]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> b{6.0, 6.0, 6.0};

    auto result = utils::solver::gmres(A, b);
    REQUIRE(result.has_value());

    CHECK(result->x(0) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(1) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(2) == Catch::Approx(1.0).epsilon(1e-8));
}

// ─────────────────────────────────────────────────────────────────────────────
// BiCGSTAB
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("bicgstab 2x2 SPD", "[solver][bicgstab]") {
    Matrix<double, 2, 2> A{{4.0, 1.0}, {1.0, 3.0}};
    Vector<double, 2> b{1.0, 2.0};

    auto result = utils::solver::bicgstab(A, b);
    REQUIRE(result.has_value());

    CHECK(result->x(0) == Catch::Approx(1.0 / 11.0).epsilon(1e-8));
    CHECK(result->x(1) == Catch::Approx(7.0 / 11.0).epsilon(1e-8));
}

TEST_CASE("bicgstab 3x3 non-symmetric", "[solver][bicgstab]") {
    Matrix<double, 3, 3> A{
        {3.0, 1.0, 0.0}, {0.0, 3.0, 1.0}, {0.0, 0.0, 3.0}};
    Vector<double, 3> b{4.0, 4.0, 3.0};

    auto result = utils::solver::bicgstab(A, b);
    REQUIRE(result.has_value());

    CHECK(result->x(0) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(1) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(2) == Catch::Approx(1.0).epsilon(1e-8));
}

TEST_CASE("bicgstab 3x3 SPD", "[solver][bicgstab]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> b{6.0, 6.0, 6.0};

    auto result = utils::solver::bicgstab(A, b);
    REQUIRE(result.has_value());

    CHECK(result->x(0) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(1) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(2) == Catch::Approx(1.0).epsilon(1e-8));
}
