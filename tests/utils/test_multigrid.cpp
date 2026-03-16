
#include <cmath>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/utils/solver/multigrid.hpp>

#include "../catch_include.hpp"

using namespace zipper;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build a 1D Laplacian matrix for n interior points.
// ─────────────────────────────────────────────────────────────────────────────

static auto make_1d_laplacian(index_type n) -> MatrixXX<double> {
    const double h2_inv =
        static_cast<double>((n + 1) * (n + 1));

    MatrixXX<double> A(n, n);
    A = expression::nullary::Constant(0.0, A.extents());
    for (index_type i = 0; i < n; ++i) {
        A(i, i) = 2.0 * h2_inv;
        if (i > 0) A(i, i - 1) = -1.0 * h2_inv;
        if (i + 1 < n) A(i, i + 1) = -1.0 * h2_inv;
    }
    return A;
}

// ─────────────────────────────────────────────────────────────────────────────
// build_1d_hierarchy
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("build_1d_hierarchy creates correct number of levels",
          "[solver][multigrid]") {
    // n = 7 = 2^3 - 1 -> can have 3 levels: 7, 3, 1.
    auto A = make_1d_laplacian(7);
    auto hierarchy = utils::solver::build_1d_hierarchy<double>(A, 3);

    REQUIRE(hierarchy.size() == 3);
    CHECK(hierarchy[0].A.rows() == 7);
    CHECK(hierarchy[1].A.rows() == 3);
    CHECK(hierarchy[2].A.rows() == 1);
}

TEST_CASE("build_1d_hierarchy restriction/prolongation sizes",
          "[solver][multigrid]") {
    auto A = make_1d_laplacian(7);
    auto hierarchy = utils::solver::build_1d_hierarchy<double>(A, 3);

    // Level 0: R is 3x7, P is 7x3.
    CHECK(hierarchy[0].R.rows() == 3);
    CHECK(hierarchy[0].R.cols() == 7);
    CHECK(hierarchy[0].P.rows() == 7);
    CHECK(hierarchy[0].P.cols() == 3);

    // Level 1: R is 1x3, P is 3x1.
    CHECK(hierarchy[1].R.rows() == 1);
    CHECK(hierarchy[1].R.cols() == 3);
    CHECK(hierarchy[1].P.rows() == 3);
    CHECK(hierarchy[1].P.cols() == 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Multigrid V-cycle solver: 1D Laplacian n=7
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("multigrid 1D Laplacian n=7", "[solver][multigrid]") {
    constexpr index_type n = 7;
    auto A = make_1d_laplacian(n);

    // RHS: constant forcing b_i = 1.
    VectorX<double> b(n);
    for (index_type i = 0; i < n; ++i) {
        b(i) = 1.0;
    }

    auto hierarchy = utils::solver::build_1d_hierarchy<double>(A, 3);
    auto result = utils::solver::multigrid(hierarchy, b);

    REQUIRE(result.has_value());
    CHECK(result->residual_norm < 1e-8);

    // Verify A*x = b.
    VectorX<double> Ax(hierarchy[0].A * result->x);
    for (index_type i = 0; i < n; ++i) {
        CHECK(Ax(i) == Catch::Approx(b(i)).epsilon(1e-6));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multigrid V-cycle solver: 1D Laplacian n=15
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("multigrid 1D Laplacian n=15", "[solver][multigrid]") {
    constexpr index_type n = 15; // 2^4 - 1
    auto A = make_1d_laplacian(n);

    VectorX<double> b(n);
    for (index_type i = 0; i < n; ++i) {
        b(i) = 1.0;
    }

    // 4 levels: 15, 7, 3, 1.
    auto hierarchy = utils::solver::build_1d_hierarchy<double>(A, 4);
    auto result = utils::solver::multigrid(hierarchy, b);

    REQUIRE(result.has_value());
    CHECK(result->residual_norm < 1e-8);

    VectorX<double> Ax(hierarchy[0].A * result->x);
    for (index_type i = 0; i < n; ++i) {
        CHECK(Ax(i) == Catch::Approx(b(i)).epsilon(1e-6));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multigrid with initial guess
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("multigrid with initial guess", "[solver][multigrid]") {
    constexpr index_type n = 7;
    auto A = make_1d_laplacian(n);

    VectorX<double> b(n);
    for (index_type i = 0; i < n; ++i) {
        b(i) = 1.0;
    }

    auto hierarchy = utils::solver::build_1d_hierarchy<double>(A, 3);

    // Start with a non-zero initial guess.
    VectorX<double> x0(n);
    for (index_type i = 0; i < n; ++i) {
        x0(i) = 0.5;
    }

    auto result = utils::solver::multigrid(hierarchy, b, x0);

    REQUIRE(result.has_value());
    CHECK(result->residual_norm < 1e-8);
}

// ─────────────────────────────────────────────────────────────────────────────
// Multigrid: 2-level hierarchy (n=3)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("multigrid 1D Laplacian n=3 two levels", "[solver][multigrid]") {
    constexpr index_type n = 3; // 2^2 - 1
    auto A = make_1d_laplacian(n);

    VectorX<double> b(n);
    b(0) = 1.0;
    b(1) = 2.0;
    b(2) = 1.0;

    // 2 levels: 3, 1.
    auto hierarchy = utils::solver::build_1d_hierarchy<double>(A, 2);

    REQUIRE(hierarchy.size() == 2);

    auto result = utils::solver::multigrid(hierarchy, b);

    REQUIRE(result.has_value());
    CHECK(result->residual_norm < 1e-8);

    VectorX<double> Ax(hierarchy[0].A * result->x);
    for (index_type i = 0; i < n; ++i) {
        CHECK(Ax(i) == Catch::Approx(b(i)).epsilon(1e-6));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multigrid: non-uniform RHS
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("multigrid 1D Laplacian non-uniform RHS", "[solver][multigrid]") {
    constexpr index_type n = 7;
    auto A = make_1d_laplacian(n);

    // Non-uniform RHS: b_i = sin(pi * i / (n+1)).
    VectorX<double> b(n);
    for (index_type i = 0; i < n; ++i) {
        b(i) = std::sin(M_PI * static_cast<double>(i + 1) /
                        static_cast<double>(n + 1));
    }

    auto hierarchy = utils::solver::build_1d_hierarchy<double>(A, 3);
    auto result = utils::solver::multigrid(hierarchy, b);

    REQUIRE(result.has_value());
    CHECK(result->residual_norm < 1e-8);

    VectorX<double> Ax(hierarchy[0].A * result->x);
    for (index_type i = 0; i < n; ++i) {
        CHECK(Ax(i) == Catch::Approx(b(i)).epsilon(1e-6));
    }
}
