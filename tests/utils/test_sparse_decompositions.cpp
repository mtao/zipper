
#include <cmath>
#include <vector>

#include <zipper/COOMatrix.hpp>
#include <zipper/CSRMatrix.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/decomposition/sparse_ldlt.hpp>
#include <zipper/utils/decomposition/sparse_llt.hpp>
#include <zipper/utils/decomposition/sparse_lu.hpp>
#include <zipper/utils/decomposition/sparse_qr.hpp>

#include "../catch_include.hpp"

using namespace zipper;

// ═══════════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Build a CSR matrix from a dense Matrix for test convenience.
template <typename T, index_type M, index_type N>
auto to_csr(const Matrix<T, M, N> &A) -> CSRMatrix<T, M, N> {
    COOMatrix<T, M, N> coo;
    for (index_type i = 0; i < A.extent(0); ++i) {
        for (index_type j = 0; j < A.extent(1); ++j) {
            if (A(i, j) != T{0}) {
                coo.emplace(i, j) = A(i, j);
            }
        }
    }
    coo.compress();
    return CSRMatrix<T, M, N>(coo);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Sparse PLU
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sparse_plu 2x2", "[sparse][decomposition][plu]") {
    Matrix<double, 2, 2> Ad{{2.0, 1.0}, {4.0, 3.0}};
    auto A = to_csr(Ad);

    auto result = utils::decomposition::sparse_plu(A);
    REQUIRE(result.has_value());

    // Verify P*A = L*U by checking that solve recovers x_true
    Vector<double, 2> x_true{1.0, 2.0};
    // b = Ad * x_true = [2+2, 4+6] = [4, 10]
    Vector<double, 2> b{4.0, 10.0};

    auto x = result->solve(b);
    REQUIRE(x.has_value());
    CHECK(x->operator()(0) == Catch::Approx(1.0).margin(1e-10));
    CHECK(x->operator()(1) == Catch::Approx(2.0).margin(1e-10));
}

TEST_CASE("sparse_plu 3x3", "[sparse][decomposition][plu]") {
    Matrix<double, 3, 3> Ad{
        {2.0, 1.0, 1.0}, {4.0, 3.0, 3.0}, {8.0, 7.0, 9.0}};
    auto A = to_csr(Ad);

    auto result = utils::decomposition::sparse_plu(A);
    REQUIRE(result.has_value());

    Vector<double, 3> x_true{1.0, 2.0, 3.0};
    Vector<double, 3> b(3);
    for (index_type i = 0; i < 3; ++i) {
        double sum = 0.0;
        for (index_type j = 0; j < 3; ++j) {
            sum += Ad(i, j) * x_true(j);
        }
        b(i) = sum;
    }

    auto x = result->solve(b);
    REQUIRE(x.has_value());
    for (index_type i = 0; i < 3; ++i) {
        CHECK(x->operator()(i) == Catch::Approx(x_true(i)).margin(1e-10));
    }
}

TEST_CASE("sparse_plu identity determinant", "[sparse][decomposition][plu]") {
    Matrix<double, 3, 3> Id{
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    auto I = to_csr(Id);

    auto result = utils::decomposition::sparse_plu(I);
    REQUIRE(result.has_value());
    CHECK(result->determinant() == Catch::Approx(1.0).margin(1e-12));
}

TEST_CASE("sparse_plu determinant 2x2", "[sparse][decomposition][plu]") {
    Matrix<double, 2, 2> Ad{{3.0, 8.0}, {4.0, 6.0}};
    auto A = to_csr(Ad);
    // det = 3*6 - 8*4 = -14

    auto result = utils::decomposition::sparse_plu(A);
    REQUIRE(result.has_value());
    CHECK(result->determinant() == Catch::Approx(-14.0).margin(1e-10));
}

TEST_CASE("sparse_plu singular fails", "[sparse][decomposition][plu]") {
    Matrix<double, 2, 2> Ad{{1.0, 2.0}, {2.0, 4.0}};
    auto A = to_csr(Ad);

    auto result = utils::decomposition::sparse_plu(A);
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().kind == utils::solver::SolverError::Kind::breakdown);
}

TEST_CASE("sparse_plu_solve convenience", "[sparse][decomposition][plu]") {
    Matrix<double, 3, 3> Ad{
        {2.0, 1.0, 1.0}, {4.0, 3.0, 3.0}, {8.0, 7.0, 9.0}};
    auto A = to_csr(Ad);
    Vector<double, 3> b{4.0, 10.0, 24.0};

    auto x = utils::decomposition::sparse_plu_solve(A, b);
    REQUIRE(x.has_value());

    // Verify A*x ≈ b
    for (index_type i = 0; i < 3; ++i) {
        double sum = 0.0;
        for (index_type j = 0; j < 3; ++j) {
            sum += Ad(i, j) * x->operator()(j);
        }
        CHECK(sum == Catch::Approx(b(i)).margin(1e-10));
    }
}

TEST_CASE("sparse_plu non-symmetric", "[sparse][decomposition][plu]") {
    // Upper triangular — PLU handles general non-symmetric matrices.
    Matrix<double, 3, 3> Ad{
        {3.0, 1.0, 0.0}, {0.0, 3.0, 1.0}, {0.0, 0.0, 3.0}};
    auto A = to_csr(Ad);
    Vector<double, 3> b{4.0, 4.0, 3.0};

    auto x = utils::decomposition::sparse_plu_solve(A, b);
    REQUIRE(x.has_value());
    CHECK(x->operator()(0) == Catch::Approx(1.0).margin(1e-10));
    CHECK(x->operator()(1) == Catch::Approx(1.0).margin(1e-10));
    CHECK(x->operator()(2) == Catch::Approx(1.0).margin(1e-10));
}

TEST_CASE("sparse_plu with coo input", "[sparse][decomposition][plu]") {
    // sparse_plu accepts anything satisfying concepts::Matrix, including COO
    COOMatrix<double, 2, 2> A;
    A.emplace(0, 0) = 2.0;
    A.emplace(0, 1) = 1.0;
    A.emplace(1, 0) = 4.0;
    A.emplace(1, 1) = 3.0;
    A.compress();

    auto result = utils::decomposition::sparse_plu(A);
    REQUIRE(result.has_value());
    CHECK(result->determinant() == Catch::Approx(2.0).margin(1e-10));
}

// ═══════════════════════════════════════════════════════════════════════════
//  Sparse LLT (Cholesky)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sparse_llt 2x2 SPD", "[sparse][decomposition][llt]") {
    // A = [[4, 2], [2, 3]] — SPD
    Matrix<double, 2, 2> Ad{{4.0, 2.0}, {2.0, 3.0}};
    auto A = to_csr(Ad);

    auto result = utils::decomposition::sparse_llt(A);
    REQUIRE(result.has_value());

    // Verify solve: x_true = [1, 1], b = [6, 5]
    Vector<double, 2> b{6.0, 5.0};
    auto x = result->solve(b);
    REQUIRE(x.has_value());
    CHECK(x->operator()(0) == Catch::Approx(1.0).margin(1e-10));
    CHECK(x->operator()(1) == Catch::Approx(1.0).margin(1e-10));
}

TEST_CASE("sparse_llt 3x3 SPD", "[sparse][decomposition][llt]") {
    // A = [[4, 1, 1], [1, 4, 1], [1, 1, 4]] — SPD
    Matrix<double, 3, 3> Ad{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    auto A = to_csr(Ad);

    auto result = utils::decomposition::sparse_llt(A);
    REQUIRE(result.has_value());

    Vector<double, 3> b{6.0, 6.0, 6.0};
    auto x = result->solve(b);
    REQUIRE(x.has_value());
    CHECK(x->operator()(0) == Catch::Approx(1.0).margin(1e-10));
    CHECK(x->operator()(1) == Catch::Approx(1.0).margin(1e-10));
    CHECK(x->operator()(2) == Catch::Approx(1.0).margin(1e-10));
}

TEST_CASE("sparse_llt identity", "[sparse][decomposition][llt]") {
    Matrix<double, 3, 3> Id{
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    auto I = to_csr(Id);

    auto result = utils::decomposition::sparse_llt(I);
    REQUIRE(result.has_value());

    // L should be identity for identity input
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(result->L(i, j) == Catch::Approx(expected).margin(1e-12));
        }
    }
}

TEST_CASE("sparse_llt non-SPD fails", "[sparse][decomposition][llt]") {
    // Not positive definite: eigenvalue < 0
    Matrix<double, 2, 2> Ad{{1.0, 3.0}, {3.0, 1.0}};
    auto A = to_csr(Ad);

    auto result = utils::decomposition::sparse_llt(A);
    REQUIRE_FALSE(result.has_value());
}

TEST_CASE("sparse_llt_solve convenience", "[sparse][decomposition][llt]") {
    Matrix<double, 3, 3> Ad{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    auto A = to_csr(Ad);
    Vector<double, 3> x_true{1.0, 2.0, 3.0};
    Vector<double, 3> b(3);
    for (index_type i = 0; i < 3; ++i) {
        double sum = 0.0;
        for (index_type j = 0; j < 3; ++j) {
            sum += Ad(i, j) * x_true(j);
        }
        b(i) = sum;
    }

    auto x = utils::decomposition::sparse_llt_solve(A, b);
    REQUIRE(x.has_value());
    for (index_type i = 0; i < 3; ++i) {
        CHECK(x->operator()(i) == Catch::Approx(x_true(i)).margin(1e-10));
    }
}

TEST_CASE("sparse_llt L*L^T reconstructs A", "[sparse][decomposition][llt]") {
    Matrix<double, 3, 3> Ad{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    auto A = to_csr(Ad);

    auto result = utils::decomposition::sparse_llt(A);
    REQUIRE(result.has_value());

    // Verify L * L^T ≈ A
    const auto &L = result->L;
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (index_type k = 0; k < 3; ++k) {
                sum += L(i, k) * L(j, k); // L * L^T
            }
            CHECK(sum == Catch::Approx(Ad(i, j)).margin(1e-10));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Sparse LDLT
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sparse_ldlt 2x2 SPD", "[sparse][decomposition][ldlt]") {
    Matrix<double, 2, 2> Ad{{4.0, 2.0}, {2.0, 3.0}};
    auto A = to_csr(Ad);

    auto result = utils::decomposition::sparse_ldlt(A);
    REQUIRE(result.has_value());

    Vector<double, 2> b{6.0, 5.0};
    auto x = result->solve(b);
    REQUIRE(x.has_value());
    CHECK(x->operator()(0) == Catch::Approx(1.0).margin(1e-10));
    CHECK(x->operator()(1) == Catch::Approx(1.0).margin(1e-10));
}

TEST_CASE("sparse_ldlt 3x3 SPD", "[sparse][decomposition][ldlt]") {
    Matrix<double, 3, 3> Ad{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    auto A = to_csr(Ad);

    auto result = utils::decomposition::sparse_ldlt(A);
    REQUIRE(result.has_value());

    Vector<double, 3> b{6.0, 6.0, 6.0};
    auto x = result->solve(b);
    REQUIRE(x.has_value());
    CHECK(x->operator()(0) == Catch::Approx(1.0).margin(1e-10));
    CHECK(x->operator()(1) == Catch::Approx(1.0).margin(1e-10));
    CHECK(x->operator()(2) == Catch::Approx(1.0).margin(1e-10));
}

TEST_CASE("sparse_ldlt identity", "[sparse][decomposition][ldlt]") {
    Matrix<double, 3, 3> Id{
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    auto I = to_csr(Id);

    auto result = utils::decomposition::sparse_ldlt(I);
    REQUIRE(result.has_value());

    // L should be identity, D should be all ones
    for (index_type i = 0; i < 3; ++i) {
        CHECK(result->D(i) == Catch::Approx(1.0).margin(1e-12));
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(result->L(i, j) == Catch::Approx(expected).margin(1e-12));
        }
    }
}

TEST_CASE("sparse_ldlt rank", "[sparse][decomposition][ldlt]") {
    Matrix<double, 3, 3> Ad{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    auto A = to_csr(Ad);

    auto result = utils::decomposition::sparse_ldlt(A);
    REQUIRE(result.has_value());
    CHECK(result->rank() == 3);
}

TEST_CASE("sparse_ldlt rank deficient", "[sparse][decomposition][ldlt]") {
    // Rank-1 PSD matrix: A = v * v^T where v = [1, 2, 3]
    // A = [[1,2,3],[2,4,6],[3,6,9]]
    Matrix<double, 3, 3> Ad{
        {1.0, 2.0, 3.0}, {2.0, 4.0, 6.0}, {3.0, 6.0, 9.0}};
    auto A = to_csr(Ad);

    auto result = utils::decomposition::sparse_ldlt(A);
    REQUIRE(result.has_value());
    CHECK(result->rank() == 1);
}

TEST_CASE("sparse_ldlt L*D*L^T reconstructs A",
          "[sparse][decomposition][ldlt]") {
    Matrix<double, 3, 3> Ad{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    auto A = to_csr(Ad);

    auto result = utils::decomposition::sparse_ldlt(A);
    REQUIRE(result.has_value());

    const auto &L = result->L;
    const auto &D = result->D;

    // Verify L * D * L^T ≈ A
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (index_type k = 0; k < 3; ++k) {
                sum += L(i, k) * D(k) * L(j, k); // L * D * L^T
            }
            CHECK(sum == Catch::Approx(Ad(i, j)).margin(1e-10));
        }
    }
}

TEST_CASE("sparse_ldlt_solve convenience", "[sparse][decomposition][ldlt]") {
    Matrix<double, 3, 3> Ad{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    auto A = to_csr(Ad);
    Vector<double, 3> x_true{1.0, 2.0, 3.0};
    Vector<double, 3> b(3);
    for (index_type i = 0; i < 3; ++i) {
        double sum = 0.0;
        for (index_type j = 0; j < 3; ++j) {
            sum += Ad(i, j) * x_true(j);
        }
        b(i) = sum;
    }

    auto x = utils::decomposition::sparse_ldlt_solve(A, b);
    REQUIRE(x.has_value());
    for (index_type i = 0; i < 3; ++i) {
        CHECK(x->operator()(i) == Catch::Approx(x_true(i)).margin(1e-10));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Sparse QR
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sparse_qr 3x3 full rank", "[sparse][decomposition][qr]") {
    Matrix<double, 3, 3> Ad{
        {1.0, 1.0, 0.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}};
    auto A = to_csr(Ad);

    auto result = utils::decomposition::sparse_qr(A);
    REQUIRE(result.has_value());

    CHECK(result->rank() == 3);
}

TEST_CASE("sparse_qr solve 3x3", "[sparse][decomposition][qr]") {
    Matrix<double, 3, 3> Ad{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    auto A = to_csr(Ad);

    Vector<double, 3> x_true{1.0, 2.0, 3.0};
    Vector<double, 3> b(3);
    for (index_type i = 0; i < 3; ++i) {
        double sum = 0.0;
        for (index_type j = 0; j < 3; ++j) {
            sum += Ad(i, j) * x_true(j);
        }
        b(i) = sum;
    }

    auto result = utils::decomposition::sparse_qr(A);
    REQUIRE(result.has_value());

    auto x = result->solve(b);
    REQUIRE(x.has_value());
    for (index_type i = 0; i < 3; ++i) {
        CHECK(x->operator()(i) == Catch::Approx(x_true(i)).margin(1e-10));
    }
}

TEST_CASE("sparse_qr solve 2x2", "[sparse][decomposition][qr]") {
    Matrix<double, 2, 2> Ad{{4.0, 1.0}, {1.0, 3.0}};
    auto A = to_csr(Ad);
    Vector<double, 2> b{1.0, 2.0};

    auto x = utils::decomposition::sparse_qr_solve(A, b);
    REQUIRE(x.has_value());
    CHECK(x->operator()(0) == Catch::Approx(1.0 / 11.0).margin(1e-10));
    CHECK(x->operator()(1) == Catch::Approx(7.0 / 11.0).margin(1e-10));
}

TEST_CASE("sparse_qr rank deficient", "[sparse][decomposition][qr]") {
    // col 2 = col 0 + col 1 => rank 2
    Matrix<double, 3, 3> Ad{
        {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}, {1.0, 1.0, 2.0}};
    auto A = to_csr(Ad);

    auto result = utils::decomposition::sparse_qr(A);
    REQUIRE(result.has_value());
    CHECK(result->rank() == 2);
}

TEST_CASE("sparse_qr rank 1", "[sparse][decomposition][qr]") {
    // All columns are multiples of [1, 2, 3]
    Matrix<double, 3, 3> Ad{
        {1.0, 2.0, 3.0}, {2.0, 4.0, 6.0}, {3.0, 6.0, 9.0}};
    auto A = to_csr(Ad);

    auto result = utils::decomposition::sparse_qr(A);
    REQUIRE(result.has_value());
    CHECK(result->rank() == 1);
}

TEST_CASE("sparse_qr zero matrix rank 0", "[sparse][decomposition][qr]") {
    Matrix<double, 3, 3> Ad{
        {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    auto A = to_csr(Ad);

    auto result = utils::decomposition::sparse_qr(A);
    REQUIRE(result.has_value());
    CHECK(result->rank() == 0);
}

TEST_CASE("sparse_qr R is upper triangular", "[sparse][decomposition][qr]") {
    Matrix<double, 3, 3> Ad{
        {1.0, 1.0, 0.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}};
    auto A = to_csr(Ad);

    auto result = utils::decomposition::sparse_qr(A);
    REQUIRE(result.has_value());

    const auto &R = result->R;
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < i; ++j) {
            CHECK(R(i, j) == Catch::Approx(0.0).margin(1e-10));
        }
    }
}

TEST_CASE("sparse_qr_solve convenience", "[sparse][decomposition][qr]") {
    Matrix<double, 3, 3> Ad{
        {3.0, 1.0, 0.0}, {0.0, 3.0, 1.0}, {0.0, 0.0, 3.0}};
    auto A = to_csr(Ad);
    Vector<double, 3> b{4.0, 4.0, 3.0};

    auto x = utils::decomposition::sparse_qr_solve(A, b);
    REQUIRE(x.has_value());
    CHECK(x->operator()(0) == Catch::Approx(1.0).margin(1e-10));
    CHECK(x->operator()(1) == Catch::Approx(1.0).margin(1e-10));
    CHECK(x->operator()(2) == Catch::Approx(1.0).margin(1e-10));
}

TEST_CASE("sparse_qr identity", "[sparse][decomposition][qr]") {
    Matrix<double, 3, 3> Id{
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    auto I = to_csr(Id);

    auto result = utils::decomposition::sparse_qr(I);
    REQUIRE(result.has_value());
    CHECK(result->rank() == 3);

    // R should be identity (or close)
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(result->R(i, j) == Catch::Approx(expected).margin(1e-10));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Cross-solver consistency
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sparse decompositions agree on SPD system",
          "[sparse][decomposition]") {
    // Same SPD system solved by all four methods
    Matrix<double, 3, 3> Ad{
        {4.0, 1.0, 1.0}, {1.0, 4.0, 1.0}, {1.0, 1.0, 4.0}};
    auto A = to_csr(Ad);
    Vector<double, 3> b{6.0, 6.0, 6.0};

    auto x_plu = utils::decomposition::sparse_plu_solve(A, b);
    auto x_llt = utils::decomposition::sparse_llt_solve(A, b);
    auto x_ldlt = utils::decomposition::sparse_ldlt_solve(A, b);
    auto x_qr = utils::decomposition::sparse_qr_solve(A, b);

    REQUIRE(x_plu.has_value());
    REQUIRE(x_llt.has_value());
    REQUIRE(x_ldlt.has_value());
    REQUIRE(x_qr.has_value());

    for (index_type i = 0; i < 3; ++i) {
        double plu_val = x_plu->operator()(i);
        CHECK(x_llt->operator()(i) == Catch::Approx(plu_val).margin(1e-10));
        CHECK(x_ldlt->operator()(i) == Catch::Approx(plu_val).margin(1e-10));
        CHECK(x_qr->operator()(i) == Catch::Approx(plu_val).margin(1e-10));
    }
}

TEST_CASE("sparse_plu reuse factorisation",
          "[sparse][decomposition][plu]") {
    Matrix<double, 3, 3> Ad{
        {2.0, 1.0, 1.0}, {4.0, 3.0, 3.0}, {8.0, 7.0, 9.0}};
    auto A = to_csr(Ad);

    auto decomp = utils::decomposition::sparse_plu(A);
    REQUIRE(decomp.has_value());

    // Solve for two different right-hand sides
    Vector<double, 3> b1{4.0, 10.0, 24.0};
    auto x1 = decomp->solve(b1);
    REQUIRE(x1.has_value());

    Vector<double, 3> b2{1.0, 3.0, 9.0};
    auto x2 = decomp->solve(b2);
    REQUIRE(x2.has_value());

    // Verify both solutions: A*x ≈ b
    for (index_type i = 0; i < 3; ++i) {
        double sum1 = 0.0, sum2 = 0.0;
        for (index_type j = 0; j < 3; ++j) {
            sum1 += Ad(i, j) * x1->operator()(j);
            sum2 += Ad(i, j) * x2->operator()(j);
        }
        CHECK(sum1 == Catch::Approx(b1(i)).margin(1e-10));
        CHECK(sum2 == Catch::Approx(b2(i)).margin(1e-10));
    }
}

TEST_CASE("sparse_plu 4x4", "[sparse][decomposition][plu]") {
    Matrix<double, 4, 4> Ad{
        {10.0, 1.0, 2.0, 1.0},
        {1.0, 10.0, 1.0, 2.0},
        {2.0, 1.0, 10.0, 1.0},
        {1.0, 2.0, 1.0, 10.0}};
    auto A = to_csr(Ad);
    Vector<double, 4> x_true{1.0, -1.0, 2.0, -2.0};
    Vector<double, 4> b(4);
    for (index_type i = 0; i < 4; ++i) {
        double sum = 0.0;
        for (index_type j = 0; j < 4; ++j) {
            sum += Ad(i, j) * x_true(j);
        }
        b(i) = sum;
    }

    auto x = utils::decomposition::sparse_plu_solve(A, b);
    REQUIRE(x.has_value());
    for (index_type i = 0; i < 4; ++i) {
        CHECK(x->operator()(i) == Catch::Approx(x_true(i)).margin(1e-10));
    }
}
