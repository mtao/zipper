
#include <algorithm>
#include <vector>

#include <zipper/COOMatrix.hpp>
#include <zipper/CSMatrix.hpp>
#include <zipper/CSRMatrix.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/sparse/add.hpp>
#include <zipper/utils/sparse/diagonal.hpp>
#include <zipper/utils/sparse/multiply.hpp>
#include <zipper/utils/sparse/transpose.hpp>

#include "../catch_include.hpp"

using namespace zipper;

// ═══════════════════════════════════════════════════════════════════════════
// Helper: build a small CSR test matrix
// ═══════════════════════════════════════════════════════════════════════════

// Returns CSR for:
//   [[1, 0, 2],
//    [0, 3, 0],
//    [4, 0, 5]]
static auto make_test_csr_3x3()
    -> CSRMatrix<double, dynamic_extent, dynamic_extent> {
    COOMatrix<double, dynamic_extent, dynamic_extent> coo(3, 3);
    coo.emplace(0, 0) = 1.0;
    coo.emplace(0, 2) = 2.0;
    coo.emplace(1, 1) = 3.0;
    coo.emplace(2, 0) = 4.0;
    coo.emplace(2, 2) = 5.0;
    coo.compress();
    return coo.to_csr();
}

// Returns CSR for:
//   [[0, 6, 0],
//    [7, 0, 8],
//    [0, 9, 0]]
static auto make_test_csr_3x3_b()
    -> CSRMatrix<double, dynamic_extent, dynamic_extent> {
    COOMatrix<double, dynamic_extent, dynamic_extent> coo(3, 3);
    coo.emplace(0, 1) = 6.0;
    coo.emplace(1, 0) = 7.0;
    coo.emplace(1, 2) = 8.0;
    coo.emplace(2, 1) = 9.0;
    coo.compress();
    return coo.to_csr();
}

// ═══════════════════════════════════════════════════════════════════════════
// Transpose tests
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sparse_transpose_csr_roundtrip", "[sparse][transpose]") {
    auto A = make_test_csr_3x3();

    // CSR -> CSC transpose.
    auto At = utils::sparse::transpose(A);

    // CSC -> CSR transpose (should recover original).
    auto Att = utils::sparse::transpose(At);

    // Verify dimensions.
    CHECK(At.extent(0) == 3);
    CHECK(At.extent(1) == 3);
    CHECK(Att.extent(0) == 3);
    CHECK(Att.extent(1) == 3);

    // Verify values: A^T^T == A.
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(Att(i, j) == Catch::Approx(A(i, j)));
        }
    }
}

TEST_CASE("sparse_transpose_values", "[sparse][transpose]") {
    auto A = make_test_csr_3x3();
    auto At = utils::sparse::transpose(A);

    // A = [[1,0,2],[0,3,0],[4,0,5]]
    // A^T = [[1,0,4],[0,3,0],[2,0,5]]
    CHECK(At(0, 0) == Catch::Approx(1.0));
    CHECK(At(0, 1) == Catch::Approx(0.0));
    CHECK(At(0, 2) == Catch::Approx(4.0));
    CHECK(At(1, 0) == Catch::Approx(0.0));
    CHECK(At(1, 1) == Catch::Approx(3.0));
    CHECK(At(1, 2) == Catch::Approx(0.0));
    CHECK(At(2, 0) == Catch::Approx(2.0));
    CHECK(At(2, 1) == Catch::Approx(0.0));
    CHECK(At(2, 2) == Catch::Approx(5.0));
}

TEST_CASE("sparse_transpose_rectangular", "[sparse][transpose]") {
    // 2x3 matrix: [[1, 0, 2], [0, 3, 0]]
    COOMatrix<double, dynamic_extent, dynamic_extent> coo(2, 3);
    coo.emplace(0, 0) = 1.0;
    coo.emplace(0, 2) = 2.0;
    coo.emplace(1, 1) = 3.0;
    coo.compress();
    auto A = coo.to_csr();

    auto At = utils::sparse::transpose(A);
    CHECK(At.extent(0) == 3);
    CHECK(At.extent(1) == 2);

    // A^T = [[1, 0], [0, 3], [2, 0]]
    CHECK(At(0, 0) == Catch::Approx(1.0));
    CHECK(At(0, 1) == Catch::Approx(0.0));
    CHECK(At(1, 0) == Catch::Approx(0.0));
    CHECK(At(1, 1) == Catch::Approx(3.0));
    CHECK(At(2, 0) == Catch::Approx(2.0));
    CHECK(At(2, 1) == Catch::Approx(0.0));
}

TEST_CASE("sparse_transpose_empty", "[sparse][transpose]") {
    COOMatrix<double, dynamic_extent, dynamic_extent> coo(3, 4);
    coo.compress();
    auto A = coo.to_csr();

    auto At = utils::sparse::transpose(A);
    CHECK(At.extent(0) == 4);
    CHECK(At.extent(1) == 3);
}

// ═══════════════════════════════════════════════════════════════════════════
// Addition tests
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sparse_add_same_pattern", "[sparse][add]") {
    auto A = make_test_csr_3x3();

    // A + A = 2*A
    auto C = utils::sparse::add(A, A);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(C(i, j) == Catch::Approx(2.0 * A(i, j)));
        }
    }
}

TEST_CASE("sparse_add_disjoint_patterns", "[sparse][add]") {
    auto A = make_test_csr_3x3();
    auto B = make_test_csr_3x3_b();

    // A and B have completely disjoint sparsity patterns.
    auto C = utils::sparse::add(A, B);

    // C should be:
    // [[1, 6, 2],
    //  [7, 3, 8],
    //  [4, 9, 5]]
    CHECK(C(0, 0) == Catch::Approx(1.0));
    CHECK(C(0, 1) == Catch::Approx(6.0));
    CHECK(C(0, 2) == Catch::Approx(2.0));
    CHECK(C(1, 0) == Catch::Approx(7.0));
    CHECK(C(1, 1) == Catch::Approx(3.0));
    CHECK(C(1, 2) == Catch::Approx(8.0));
    CHECK(C(2, 0) == Catch::Approx(4.0));
    CHECK(C(2, 1) == Catch::Approx(9.0));
    CHECK(C(2, 2) == Catch::Approx(5.0));
}

TEST_CASE("sparse_add_with_scalars", "[sparse][add]") {
    auto A = make_test_csr_3x3();
    auto B = make_test_csr_3x3();

    // 2*A - A = A
    auto C = utils::sparse::add(A, B, 2.0, -1.0);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(C(i, j) == Catch::Approx(A(i, j)));
        }
    }
}

TEST_CASE("sparse_add_cancellation", "[sparse][add]") {
    auto A = make_test_csr_3x3();

    // A - A = 0
    auto C = utils::sparse::add(A, A, 1.0, -1.0);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(C(i, j) == Catch::Approx(0.0));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SpGEMM tests
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("spgemm_identity_left", "[sparse][multiply]") {
    auto I = utils::sparse::sparse_identity<double>(3);
    auto A = make_test_csr_3x3();

    auto C = utils::sparse::spgemm(I, A);
    CHECK(C.extent(0) == 3);
    CHECK(C.extent(1) == 3);

    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(C(i, j) == Catch::Approx(A(i, j)));
        }
    }
}

TEST_CASE("spgemm_identity_right", "[sparse][multiply]") {
    auto A = make_test_csr_3x3();
    auto I = utils::sparse::sparse_identity<double>(3);

    auto C = utils::sparse::spgemm(A, I);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(C(i, j) == Catch::Approx(A(i, j)));
        }
    }
}

TEST_CASE("spgemm_known_product", "[sparse][multiply]") {
    auto A = make_test_csr_3x3();

    // Compute A * A and verify against dense product.
    // A = [[1,0,2],[0,3,0],[4,0,5]]
    // A^2 = [[1*1+2*4, 0, 1*2+2*5], [0, 9, 0], [4*1+5*4, 0, 4*2+5*5]]
    //     = [[9, 0, 12], [0, 9, 0], [24, 0, 33]]
    auto C = utils::sparse::spgemm(A, A);

    CHECK(C(0, 0) == Catch::Approx(9.0));
    CHECK(C(0, 1) == Catch::Approx(0.0));
    CHECK(C(0, 2) == Catch::Approx(12.0));
    CHECK(C(1, 0) == Catch::Approx(0.0));
    CHECK(C(1, 1) == Catch::Approx(9.0));
    CHECK(C(1, 2) == Catch::Approx(0.0));
    CHECK(C(2, 0) == Catch::Approx(24.0));
    CHECK(C(2, 1) == Catch::Approx(0.0));
    CHECK(C(2, 2) == Catch::Approx(33.0));
}

TEST_CASE("spgemm_rectangular", "[sparse][multiply]") {
    // A: 2x3, B: 3x2
    COOMatrix<double, dynamic_extent, dynamic_extent> coo_a(2, 3);
    coo_a.emplace(0, 0) = 1.0;
    coo_a.emplace(0, 2) = 2.0;
    coo_a.emplace(1, 1) = 3.0;
    coo_a.compress();
    auto A = coo_a.to_csr();

    COOMatrix<double, dynamic_extent, dynamic_extent> coo_b(3, 2);
    coo_b.emplace(0, 0) = 4.0;
    coo_b.emplace(1, 1) = 5.0;
    coo_b.emplace(2, 0) = 6.0;
    coo_b.compress();
    auto B = coo_b.to_csr();

    // C = A * B = [[1*4+2*6, 0], [0, 3*5]] = [[16, 0], [0, 15]]
    auto C = utils::sparse::spgemm(A, B);
    CHECK(C.extent(0) == 2);
    CHECK(C.extent(1) == 2);
    CHECK(C(0, 0) == Catch::Approx(16.0));
    CHECK(C(0, 1) == Catch::Approx(0.0));
    CHECK(C(1, 0) == Catch::Approx(0.0));
    CHECK(C(1, 1) == Catch::Approx(15.0));
}

TEST_CASE("spgemm_vs_dense", "[sparse][multiply]") {
    // Compare sparse product with dense product.
    auto A = make_test_csr_3x3();
    auto B = make_test_csr_3x3_b();

    auto C_sparse = utils::sparse::spgemm(A, B);

    // Dense reference: A_dense * B_dense.
    Matrix<double, 3, 3> A_dense{{{1, 0, 2}, {0, 3, 0}, {4, 0, 5}}};
    Matrix<double, 3, 3> B_dense{{{0, 6, 0}, {7, 0, 8}, {0, 9, 0}}};
    auto C_dense = Matrix<double, 3, 3>(A_dense * B_dense);

    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(C_sparse(i, j) == Catch::Approx(C_dense(i, j)));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Diagonal tests
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sparse_extract_diagonal", "[sparse][diagonal]") {
    auto A = make_test_csr_3x3();

    // A = [[1,0,2],[0,3,0],[4,0,5]], diagonal = [1, 3, 5]
    auto diag = utils::sparse::extract_diagonal(A);
    REQUIRE(diag.extent(0) == 3);
    CHECK(diag(0) == Catch::Approx(1.0));
    CHECK(diag(1) == Catch::Approx(3.0));
    CHECK(diag(2) == Catch::Approx(5.0));
}

TEST_CASE("sparse_extract_diagonal_identity", "[sparse][diagonal]") {
    auto I = utils::sparse::sparse_identity<double>(4);
    auto diag = utils::sparse::extract_diagonal(I);
    REQUIRE(diag.extent(0) == 4);
    for (index_type i = 0; i < 4; ++i) {
        CHECK(diag(i) == Catch::Approx(1.0));
    }
}

TEST_CASE("sparse_extract_diagonal_missing", "[sparse][diagonal]") {
    // Off-diagonal only: [[0, 1], [2, 0]]
    COOMatrix<double, dynamic_extent, dynamic_extent> coo(2, 2);
    coo.emplace(0, 1) = 1.0;
    coo.emplace(1, 0) = 2.0;
    coo.compress();
    auto A = coo.to_csr();

    auto diag = utils::sparse::extract_diagonal(A);
    REQUIRE(diag.extent(0) == 2);
    CHECK(diag(0) == Catch::Approx(0.0));
    CHECK(diag(1) == Catch::Approx(0.0));
}

TEST_CASE("sparse_from_diagonal", "[sparse][diagonal]") {
    Vector<double, 3> d{2.0, 0.0, 5.0};
    auto D = utils::sparse::from_diagonal(d);

    CHECK(D.extent(0) == 3);
    CHECK(D.extent(1) == 3);

    // Should be: [[2, 0, 0], [0, 0, 0], [0, 0, 5]]
    CHECK(D(0, 0) == Catch::Approx(2.0));
    CHECK(D(0, 1) == Catch::Approx(0.0));
    CHECK(D(0, 2) == Catch::Approx(0.0));
    CHECK(D(1, 0) == Catch::Approx(0.0));
    CHECK(D(1, 1) == Catch::Approx(0.0));
    CHECK(D(1, 2) == Catch::Approx(0.0));
    CHECK(D(2, 0) == Catch::Approx(0.0));
    CHECK(D(2, 1) == Catch::Approx(0.0));
    CHECK(D(2, 2) == Catch::Approx(5.0));
}

TEST_CASE("sparse_from_diagonal_roundtrip", "[sparse][diagonal]") {
    Vector<double, 4> d{1.0, 2.0, 3.0, 4.0};
    auto D = utils::sparse::from_diagonal(d);
    auto d2 = utils::sparse::extract_diagonal(D);

    REQUIRE(d2.extent(0) == 4);
    for (index_type i = 0; i < 4; ++i) {
        CHECK(d2(i) == Catch::Approx(d(i)));
    }
}

TEST_CASE("sparse_identity", "[sparse][diagonal]") {
    auto I = utils::sparse::sparse_identity<double>(3);

    CHECK(I.extent(0) == 3);
    CHECK(I.extent(1) == 3);

    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(I(i, j) == Catch::Approx(expected));
        }
    }
}

TEST_CASE("sparse_identity_matvec", "[sparse][diagonal]") {
    auto I = utils::sparse::sparse_identity<double>(3);
    Vector<double, 3> x{1.0, 2.0, 3.0};

    auto y = Vector<double, 3>(I * x);
    CHECK(y(0) == Catch::Approx(1.0));
    CHECK(y(1) == Catch::Approx(2.0));
    CHECK(y(2) == Catch::Approx(3.0));
}
