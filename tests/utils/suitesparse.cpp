/// @file suitesparse.cpp
/// @brief Tests for SuiteSparse sparse direct solver bindings.
///
/// These tests are only compiled when ZIPPER_HAS_SUITESPARSE is defined.
/// When SuiteSparse is not available, a single placeholder test passes
/// to keep the test suite green.

#include "../catch_include.hpp"

#ifdef ZIPPER_HAS_SUITESPARSE

#include <algorithm>
#include <cmath>
#include <zipper/CSMatrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/suitesparse/cholmod.hpp>
#include <zipper/utils/suitesparse/spqr.hpp>
#include <zipper/utils/suitesparse/umfpack.hpp>

using namespace zipper;
using namespace zipper::utils::suitesparse;

// ════════════════════════════════════════════════════════════════════════
// Helper: build a small CSC SPD matrix (3x3)
//
//   A = [  4  -1   0 ]
//       [ -1   4  -1 ]
//       [  0  -1   4 ]
//
// This is tridiagonal SPD with eigenvalues in (2, 6).
// ════════════════════════════════════════════════════════════════════════

static auto make_spd_3x3() {
  // Build in COO and convert to CSC.
  using CSC = CSMatrix<double, 3, 3, storage::layout_left>;
  CSC A;
  auto &e = A.expression();

  // Row 0
  e.emplace(0, 0) = 4.0;
  e.emplace(0, 1) = -1.0;

  // Row 1
  e.emplace(1, 0) = -1.0;
  e.emplace(1, 1) = 4.0;
  e.emplace(1, 2) = -1.0;

  // Row 2
  e.emplace(2, 1) = -1.0;
  e.emplace(2, 2) = 4.0;

  e.compress();
  return A;
}

// ════════════════════════════════════════════════════════════════════════
// Helper: build a general (non-symmetric) 3x3 CSC matrix
//
//   A = [ 2  3  0 ]
//       [ 1  0  5 ]
//       [ 0  4  6 ]
// ════════════════════════════════════════════════════════════════════════

static auto make_general_3x3() {
  using CSC = CSMatrix<double, 3, 3, storage::layout_left>;
  CSC A;
  auto &e = A.expression();

  e.emplace(0, 0) = 2.0;
  e.emplace(0, 1) = 3.0;
  e.emplace(1, 0) = 1.0;
  e.emplace(1, 2) = 5.0;
  e.emplace(2, 1) = 4.0;
  e.emplace(2, 2) = 6.0;

  e.compress();
  return A;
}

// ════════════════════════════════════════════════════════════════════════
// CHOLMOD tests
// ════════════════════════════════════════════════════════════════════════

TEST_CASE("cholmod_solve_spd_3x3", "[suitesparse][cholmod]") {
  auto A = make_spd_3x3();
  Vector<double, 3> b{1.0, 2.0, 3.0};

  auto result = cholmod_solve(A, b);
  REQUIRE(result.has_value());

  auto &x = *result;

  // Verify Ax ≈ b by computing residual manually.
  // A*x: row 0 = 4*x0 - x1, row 1 = -x0 + 4*x1 - x2, row 2 = -x1 + 4*x2
  double r0 = 4 * x(0) - x(1) - b(0);
  double r1 = -x(0) + 4 * x(1) - x(2) - b(1);
  double r2 = -x(1) + 4 * x(2) - b(2);
  double rnorm = std::sqrt(r0 * r0 + r1 * r1 + r2 * r2);

  CHECK(rnorm < 1e-12);
}

TEST_CASE("cholmod_factor_and_reuse", "[suitesparse][cholmod]") {
  auto A = make_spd_3x3();

  auto factored = cholmod_factor(A);
  REQUIRE(factored.has_value());

  // Solve with two different RHS vectors.
  {
    Vector<double, 3> b1{1.0, 0.0, 0.0};
    auto x1 = factored->solve(b1);
    REQUIRE(x1.has_value());
    // Just check that it runs without error.
    CHECK(x1->rows() == 3);
  }
  {
    Vector<double, 3> b2{0.0, 1.0, 0.0};
    auto x2 = factored->solve(b2);
    REQUIRE(x2.has_value());
    CHECK(x2->rows() == 3);
  }
}

// ════════════════════════════════════════════════════════════════════════
// UMFPACK tests
// ════════════════════════════════════════════════════════════════════════

TEST_CASE("umfpack_solve_general_3x3", "[suitesparse][umfpack]") {
  auto A = make_general_3x3();
  Vector<double, 3> b{1.0, 2.0, 3.0};

  auto result = umfpack_solve(A, b);
  REQUIRE(result.has_value());

  auto &x = *result;

  // Verify Ax ≈ b
  double r0 = 2 * x(0) + 3 * x(1) - b(0);
  double r1 = x(0) + 5 * x(2) - b(1);
  double r2 = 4 * x(1) + 6 * x(2) - b(2);
  double rnorm = std::sqrt(r0 * r0 + r1 * r1 + r2 * r2);

  CHECK(rnorm < 1e-12);
}

TEST_CASE("umfpack_factor_and_reuse", "[suitesparse][umfpack]") {
  auto A = make_general_3x3();

  auto factored = umfpack_factor(A);
  REQUIRE(factored.has_value());

  Vector<double, 3> b1{1.0, 0.0, 0.0};
  auto x1 = factored->solve(b1);
  REQUIRE(x1.has_value());
  CHECK(x1->rows() == 3);

  Vector<double, 3> b2{0.0, 0.0, 1.0};
  auto x2 = factored->solve(b2);
  REQUIRE(x2.has_value());
  CHECK(x2->rows() == 3);
}

// ════════════════════════════════════════════════════════════════════════
// SPQR tests
// ════════════════════════════════════════════════════════════════════════

TEST_CASE("spqr_solve_general_3x3", "[suitesparse][spqr]") {
  auto A = make_general_3x3();
  Vector<double, 3> b{1.0, 2.0, 3.0};

  auto result = spqr_solve(A, b);
  REQUIRE(result.has_value());

  auto &x = *result;

  // Verify Ax ≈ b
  double r0 = 2 * x(0) + 3 * x(1) - b(0);
  double r1 = x(0) + 5 * x(2) - b(1);
  double r2 = 4 * x(1) + 6 * x(2) - b(2);
  double rnorm = std::sqrt(r0 * r0 + r1 * r1 + r2 * r2);

  CHECK(rnorm < 1e-12);
}

TEST_CASE("spqr_solve_spd_3x3", "[suitesparse][spqr]") {
  // SPQR can also solve SPD systems (just less efficiently than CHOLMOD).
  auto A = make_spd_3x3();
  Vector<double, 3> b{1.0, 2.0, 3.0};

  auto result = spqr_solve(A, b);
  REQUIRE(result.has_value());

  auto &x = *result;

  double r0 = 4 * x(0) - x(1) - b(0);
  double r1 = -x(0) + 4 * x(1) - x(2) - b(1);
  double r2 = -x(1) + 4 * x(2) - b(2);
  double rnorm = std::sqrt(r0 * r0 + r1 * r1 + r2 * r2);

  CHECK(rnorm < 1e-12);
}

TEST_CASE("spqr_factor_and_reuse", "[suitesparse][spqr]") {
  auto A = make_general_3x3();

  auto factored = spqr_factor(A);
  REQUIRE(factored.has_value());

  // Solve with two different RHS vectors using stored factorization.
  {
    Vector<double, 3> b1{1.0, 0.0, 0.0};
    auto x1 = factored->solve(b1);
    REQUIRE(x1.has_value());
    CHECK(x1->rows() == 3);
  }
  {
    Vector<double, 3> b2{0.0, 1.0, 0.0};
    auto x2 = factored->solve(b2);
    REQUIRE(x2.has_value());
    CHECK(x2->rows() == 3);
  }
}

TEST_CASE("spqr_rank_full_rank", "[suitesparse][spqr]") {
  auto A = make_general_3x3();

  auto factored = spqr_factor(A);
  REQUIRE(factored.has_value());

  // The 3x3 general matrix is full rank.
  CHECK(factored->rank() == 3);
}

TEST_CASE("spqr_rank_convenience", "[suitesparse][spqr]") {
  auto A = make_general_3x3();
  CHECK(spqr_rank(A) == 3);
}

TEST_CASE("spqr_permutation", "[suitesparse][spqr]") {
  auto A = make_general_3x3();

  auto factored = spqr_factor(A);
  REQUIRE(factored.has_value());

  auto perm = factored->permutation();
  // Permutation is either empty (identity) or a valid permutation of {0,1,2}.
  if (!perm.empty()) {
    REQUIRE(perm.size() == 3);
    auto sorted = perm;
    std::sort(sorted.begin(), sorted.end());
    CHECK(sorted[0] == 0);
    CHECK(sorted[1] == 1);
    CHECK(sorted[2] == 2);
  }
}

// ════════════════════════════════════════════════════════════════════════
// Dynamic extent tests (runtime-sized matrices)
// ════════════════════════════════════════════════════════════════════════

TEST_CASE("umfpack_solve_dynamic_extents", "[suitesparse][umfpack][dynamic]") {
  using CSCX = CSMatrix<double, dynamic_extent, dynamic_extent,
                        storage::layout_left>;
  CSCX A;
  auto &e = A.expression();

  // Same general matrix but with dynamic extents
  e.emplace(0, 0) = 2.0;
  e.emplace(0, 1) = 3.0;
  e.emplace(1, 0) = 1.0;
  e.emplace(1, 2) = 5.0;
  e.emplace(2, 1) = 4.0;
  e.emplace(2, 2) = 6.0;
  e.compress();

  VectorX<double> b = {1.0, 2.0, 3.0};

  auto result = umfpack_solve(A, b);
  REQUIRE(result.has_value());
  CHECK(result->rows() == 3);
}

#else // !ZIPPER_HAS_SUITESPARSE

TEST_CASE("suitesparse_not_available", "[suitesparse]") {
  // SuiteSparse is not installed. This test just verifies the build
  // works correctly without it.
  SUCCEED("SuiteSparse not available — tests skipped");
}

#endif // ZIPPER_HAS_SUITESPARSE
