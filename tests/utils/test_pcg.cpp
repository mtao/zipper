
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/concepts/Preconditioner.hpp>
#include <zipper/utils/solver/pcg.hpp>
#include <zipper/utils/solver/conjugate_gradient.hpp>
#include <zipper/utils/solver/preconditioner/jacobi_preconditioner.hpp>
#include <zipper/utils/solver/preconditioner/ssor_preconditioner.hpp>

#include "../catch_include.hpp"

using namespace zipper;

// ─────────────────────────────────────────────────────────────────────────────
// Static assertions: preconditioners satisfy the Preconditioner concept
// ─────────────────────────────────────────────────────────────────────────────

using JacobiP3 = utils::solver::JacobiPreconditioner<double, 3>;
using JacobiPX = utils::solver::JacobiPreconditioner<double, dynamic_extent>;
using SSORP3 = utils::solver::SSORPreconditioner<double, 3>;
using SSORPX = utils::solver::SSORPreconditioner<double, dynamic_extent>;

static_assert(concepts::Preconditioner<JacobiP3>);
static_assert(concepts::Preconditioner<JacobiPX>);
static_assert(concepts::Preconditioner<SSORP3>);
static_assert(concepts::Preconditioner<SSORPX>);

// Negative: plain types do not satisfy Preconditioner.
static_assert(!concepts::Preconditioner<int>);
static_assert(!concepts::Preconditioner<double>);
static_assert(!concepts::Preconditioner<Matrix<double, 3, 3>>);

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

// ─────────────────────────────────────────────────────────────────────────────
// PCG with Jacobi preconditioner
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("pcg jacobi 2x2 SPD", "[solver][pcg][jacobi]") {
    Matrix<double, 2, 2> A{{4.0, 1.0}, {1.0, 3.0}};
    Vector<double, 2> b{1.0, 2.0};

    utils::solver::JacobiPreconditioner<double, 2> precond(A);

    auto result =
        utils::solver::preconditioned_conjugate_gradient(A, b, precond);
    REQUIRE(result.has_value());

    CHECK(result->x(0) == Catch::Approx(1.0 / 11.0).epsilon(1e-8));
    CHECK(result->x(1) == Catch::Approx(7.0 / 11.0).epsilon(1e-8));
    CHECK(result->residual_norm < 1e-10);
}

TEST_CASE("pcg jacobi 3x3 SPD", "[solver][pcg][jacobi]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> b{6.0, 6.0, 6.0};

    utils::solver::JacobiPreconditioner<double, 3> precond(A);

    auto result =
        utils::solver::preconditioned_conjugate_gradient(A, b, precond);
    REQUIRE(result.has_value());

    CHECK(result->x(0) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(1) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(2) == Catch::Approx(1.0).epsilon(1e-8));
}

TEST_CASE("pcg jacobi with initial guess", "[solver][pcg][jacobi]") {
    Matrix<double, 2, 2> A{{4.0, 1.0}, {1.0, 3.0}};
    Vector<double, 2> b{1.0, 2.0};
    Vector<double, 2> x0{0.1, 0.6}; // close to true solution

    utils::solver::JacobiPreconditioner<double, 2> precond(A);

    auto result =
        utils::solver::preconditioned_conjugate_gradient(A, b, precond, x0);
    REQUIRE(result.has_value());

    CHECK(result->x(0) == Catch::Approx(1.0 / 11.0).epsilon(1e-8));
    CHECK(result->x(1) == Catch::Approx(7.0 / 11.0).epsilon(1e-8));
}

// ─────────────────────────────────────────────────────────────────────────────
// PCG with SSOR preconditioner
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("pcg ssor 2x2 SPD", "[solver][pcg][ssor]") {
    Matrix<double, 2, 2> A{{4.0, 1.0}, {1.0, 3.0}};
    Vector<double, 2> b{1.0, 2.0};

    utils::solver::SSORPreconditioner<double, 2> precond(A, 1.0);

    auto result =
        utils::solver::preconditioned_conjugate_gradient(A, b, precond);
    REQUIRE(result.has_value());

    CHECK(result->x(0) == Catch::Approx(1.0 / 11.0).epsilon(1e-8));
    CHECK(result->x(1) == Catch::Approx(7.0 / 11.0).epsilon(1e-8));
}

TEST_CASE("pcg ssor 3x3 SPD", "[solver][pcg][ssor]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> b{6.0, 6.0, 6.0};

    utils::solver::SSORPreconditioner<double, 3> precond(A, 1.2);

    auto result =
        utils::solver::preconditioned_conjugate_gradient(A, b, precond);
    REQUIRE(result.has_value());

    CHECK(result->x(0) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(1) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(2) == Catch::Approx(1.0).epsilon(1e-8));
}

TEST_CASE("pcg ssor omega=1 is symmetric Gauss-Seidel", "[solver][pcg][ssor]") {
    Matrix<double, 3, 3> A{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    Vector<double, 3> b{6.0, 6.0, 6.0};

    utils::solver::SSORPreconditioner<double, 3> precond(A, 1.0);

    auto result =
        utils::solver::preconditioned_conjugate_gradient(A, b, precond);
    REQUIRE(result.has_value());

    CHECK(result->x(0) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(1) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(2) == Catch::Approx(1.0).epsilon(1e-8));
}

// ─────────────────────────────────────────────────────────────────────────────
// PCG converges faster than unpreconditioned CG (on ill-conditioned system)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("pcg jacobi converges no slower than CG",
          "[solver][pcg][jacobi]") {
    // A moderately ill-conditioned 3x3 SPD system.
    Matrix<double, 3, 3> A{
        {10.0, 1.0, 1.0}, {1.0, 2.0, 0.5}, {1.0, 0.5, 5.0}};
    Vector<double, 3> b{12.0, 3.5, 6.5};

    utils::solver::JacobiPreconditioner<double, 3> precond(A);

    auto pcg_result =
        utils::solver::preconditioned_conjugate_gradient(A, b, precond);
    auto cg_result = utils::solver::conjugate_gradient(A, b);

    REQUIRE(pcg_result.has_value());
    REQUIRE(cg_result.has_value());

    // PCG should converge in no more iterations than CG.
    CHECK(pcg_result->iterations <= cg_result->iterations);
}

// ─────────────────────────────────────────────────────────────────────────────
// Dynamic-extent PCG
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("pcg jacobi dynamic extent", "[solver][pcg][jacobi][dynamic]") {
    MatrixXX<double> A(3, 3);
    A(0, 0) = 4.0; A(0, 1) = 1.0; A(0, 2) = 1.0;
    A(1, 0) = 1.0; A(1, 1) = 4.0; A(1, 2) = 1.0;
    A(2, 0) = 1.0; A(2, 1) = 1.0; A(2, 2) = 4.0;

    VectorX<double> b(3);
    b(0) = 6.0; b(1) = 6.0; b(2) = 6.0;

    utils::solver::JacobiPreconditioner<double> precond(A);

    auto result =
        utils::solver::preconditioned_conjugate_gradient(A, b, precond);
    REQUIRE(result.has_value());

    CHECK(result->x(0) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(1) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(result->x(2) == Catch::Approx(1.0).epsilon(1e-8));
}

// ─────────────────────────────────────────────────────────────────────────────
// Larger system: 1D Laplacian
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("pcg jacobi 1D Laplacian 7x7", "[solver][pcg][jacobi]") {
    // 1D Laplacian with n=7 interior points: tridiagonal -1, 2, -1
    // scaled by (n+1)^2 = 64.
    constexpr index_type n = 7;
    constexpr double h2_inv = static_cast<double>((n + 1) * (n + 1));

    MatrixXX<double> A(n, n);
    A = expression::nullary::Constant(0.0, A.extents());
    for (index_type i = 0; i < n; ++i) {
        A(i, i) = 2.0 * h2_inv;
        if (i > 0) A(i, i - 1) = -1.0 * h2_inv;
        if (i + 1 < n) A(i, i + 1) = -1.0 * h2_inv;
    }

    // RHS: b_i = 1 for all i (constant forcing).
    VectorX<double> b(n);
    for (index_type i = 0; i < n; ++i) {
        b(i) = 1.0;
    }

    utils::solver::JacobiPreconditioner<double> precond(A);

    auto result =
        utils::solver::preconditioned_conjugate_gradient(A, b, precond);
    REQUIRE(result.has_value());
    CHECK(result->residual_norm < 1e-8);

    // Verify A*x = b.
    VectorX<double> Ax(A * result->x);
    for (index_type i = 0; i < n; ++i) {
        CHECK(Ax(i) == Catch::Approx(b(i)).epsilon(1e-6));
    }
}
