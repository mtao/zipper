
#include <cmath>
#include <vector>

#include <zipper/COOMatrix.hpp>
#include <zipper/CSRMatrix.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/sparse/multiply.hpp>
#include <zipper/sparse/transpose.hpp>

#include "../catch_include.hpp"

using namespace zipper;

// ═══════════════════════════════════════════════════════════════════════════
//  sparse_transpose — COO
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sparse_transpose coo identity", "[sparse][transpose]") {
    // 3x3 identity — transpose should be the same.
    std::vector<SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{1, 1}, 1.0}, {{2, 2}, 1.0}};
    COOMatrix<double, 3, 3> I(entries);

    auto It = sparse::sparse_transpose(I);
    static_assert(std::is_same_v<decltype(It), COOMatrix<double, 3, 3>>);

    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(It(i, j) == I(i, j));
        }
    }
}

TEST_CASE("sparse_transpose coo general", "[sparse][transpose]") {
    // | 1  0  2 |
    // | 0  3  0 |
    // | 4  0  5 |
    COOMatrix<double, 3, 3> A;
    A.emplace(0, 0) = 1.0;
    A.emplace(0, 2) = 2.0;
    A.emplace(1, 1) = 3.0;
    A.emplace(2, 0) = 4.0;
    A.emplace(2, 2) = 5.0;
    A.compress();

    auto At = sparse::sparse_transpose(A);

    // At should be:
    // | 1  0  4 |
    // | 0  3  0 |
    // | 2  0  5 |
    CHECK(At(0, 0) == 1.0);
    CHECK(At(0, 1) == 0.0);
    CHECK(At(0, 2) == 4.0);
    CHECK(At(1, 0) == 0.0);
    CHECK(At(1, 1) == 3.0);
    CHECK(At(1, 2) == 0.0);
    CHECK(At(2, 0) == 2.0);
    CHECK(At(2, 1) == 0.0);
    CHECK(At(2, 2) == 5.0);
}

TEST_CASE("sparse_transpose coo rectangular", "[sparse][transpose]") {
    // 2x3 matrix:
    // | 1  0  2 |
    // | 0  3  0 |
    COOMatrix<double, 2, 3> A;
    A.emplace(0, 0) = 1.0;
    A.emplace(0, 2) = 2.0;
    A.emplace(1, 1) = 3.0;
    A.compress();

    auto At = sparse::sparse_transpose(A);
    static_assert(std::is_same_v<decltype(At), COOMatrix<double, 3, 2>>);

    CHECK(At.extent(0) == 3);
    CHECK(At.extent(1) == 2);
    CHECK(At(0, 0) == 1.0);
    CHECK(At(0, 1) == 0.0);
    CHECK(At(1, 0) == 0.0);
    CHECK(At(1, 1) == 3.0);
    CHECK(At(2, 0) == 2.0);
    CHECK(At(2, 1) == 0.0);
}

TEST_CASE("sparse_transpose coo double transpose roundtrip",
          "[sparse][transpose]") {
    COOMatrix<double, 3, 3> A;
    A.emplace(0, 0) = 1.0;
    A.emplace(0, 2) = 2.0;
    A.emplace(1, 1) = 3.0;
    A.emplace(2, 0) = 4.0;
    A.emplace(2, 2) = 5.0;
    A.compress();

    auto Att = sparse::sparse_transpose(sparse::sparse_transpose(A));
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(Att(i, j) == A(i, j));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  sparse_transpose — CSR
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sparse_transpose csr general", "[sparse][transpose]") {
    std::vector<SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{0, 2}, 2.0},
        {{1, 1}, 3.0},
        {{2, 0}, 4.0}, {{2, 2}, 5.0}};
    CSRMatrix<double, 3, 3> A(entries);

    auto At = sparse::sparse_transpose(A);
    static_assert(std::is_same_v<decltype(At), CSRMatrix<double, 3, 3>>);

    CHECK(At(0, 0) == 1.0);
    CHECK(At(0, 2) == 4.0);
    CHECK(At(1, 1) == 3.0);
    CHECK(At(2, 0) == 2.0);
    CHECK(At(2, 2) == 5.0);
    CHECK(At(0, 1) == 0.0);
}

TEST_CASE("sparse_transpose_to_coo csr", "[sparse][transpose]") {
    std::vector<SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{0, 2}, 2.0},
        {{1, 1}, 3.0}};
    CSRMatrix<double, 2, 3> A(entries);

    auto At_coo = sparse::sparse_transpose_to_coo(A);
    static_assert(std::is_same_v<decltype(At_coo), COOMatrix<double, 3, 2>>);

    CHECK(At_coo(0, 0) == 1.0);
    CHECK(At_coo(1, 1) == 3.0);
    CHECK(At_coo(2, 0) == 2.0);
}

TEST_CASE("sparse_transpose csr rectangular", "[sparse][transpose]") {
    std::vector<SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{0, 2}, 2.0},
        {{1, 1}, 3.0}};
    CSRMatrix<double, 2, 3> A(entries);

    auto At = sparse::sparse_transpose(A);
    static_assert(std::is_same_v<decltype(At), CSRMatrix<double, 3, 2>>);

    CHECK(At.extent(0) == 3);
    CHECK(At.extent(1) == 2);
    CHECK(At(0, 0) == 1.0);
    CHECK(At(1, 1) == 3.0);
    CHECK(At(2, 0) == 2.0);
}

TEST_CASE("sparse_transpose csr double transpose roundtrip",
          "[sparse][transpose]") {
    std::vector<SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{0, 2}, 2.0},
        {{1, 1}, 3.0},
        {{2, 0}, 4.0}, {{2, 2}, 5.0}};
    CSRMatrix<double, 3, 3> A(entries);

    auto Att = sparse::sparse_transpose(sparse::sparse_transpose(A));
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            CHECK(Att(i, j) == A(i, j));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  sparse_multiply — CSR * CSR
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sparse_multiply identity", "[sparse][multiply]") {
    // I * I = I
    std::vector<SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{1, 1}, 1.0}, {{2, 2}, 1.0}};
    CSRMatrix<double, 3, 3> I(entries);

    auto C = sparse::sparse_multiply(I, I);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(C(i, j) == expected);
        }
    }
}

TEST_CASE("sparse_multiply diagonal", "[sparse][multiply]") {
    // diag(2, 3, 4) * diag(5, 6, 7) = diag(10, 18, 28)
    std::vector<SparseEntry<double, 2>> a_entries = {
        {{0, 0}, 2.0}, {{1, 1}, 3.0}, {{2, 2}, 4.0}};
    std::vector<SparseEntry<double, 2>> b_entries = {
        {{0, 0}, 5.0}, {{1, 1}, 6.0}, {{2, 2}, 7.0}};
    CSRMatrix<double, 3, 3> A(a_entries);
    CSRMatrix<double, 3, 3> B(b_entries);

    auto C = sparse::sparse_multiply(A, B);
    CHECK(C(0, 0) == 10.0);
    CHECK(C(1, 1) == 18.0);
    CHECK(C(2, 2) == 28.0);
    CHECK(C(0, 1) == 0.0);
    CHECK(C(1, 0) == 0.0);
}

TEST_CASE("sparse_multiply general 3x3", "[sparse][multiply]") {
    // A = | 1  0  2 |   B = | 1  0 |
    //     | 0  3  0 |       | 0  1 |
    //     | 4  0  5 |       | 2  0 |
    //
    // C = A * B = | 1+0+4    0+0+0 |   | 5  0 |
    //             | 0+0+0    0+3+0 | = | 0  3 |
    //             | 4+0+10   0+0+0 |   | 14 0 |
    std::vector<SparseEntry<double, 2>> a_entries = {
        {{0, 0}, 1.0}, {{0, 2}, 2.0},
        {{1, 1}, 3.0},
        {{2, 0}, 4.0}, {{2, 2}, 5.0}};
    std::vector<SparseEntry<double, 2>> b_entries = {
        {{0, 0}, 1.0}, {{1, 1}, 1.0}, {{2, 0}, 2.0}};
    CSRMatrix<double, 3, 3> A(a_entries);
    CSRMatrix<double, 3, 2> B(b_entries);

    auto C = sparse::sparse_multiply(A, B);
    static_assert(std::is_same_v<decltype(C), CSRMatrix<double, 3, 2>>);

    CHECK(C(0, 0) == 5.0);
    CHECK(C(0, 1) == 0.0);
    CHECK(C(1, 0) == 0.0);
    CHECK(C(1, 1) == 3.0);
    CHECK(C(2, 0) == 14.0);
    CHECK(C(2, 1) == 0.0);
}

TEST_CASE("sparse_multiply agrees with dense", "[sparse][multiply]") {
    // A = | 1  2 |   B = | 5  6 |
    //     | 3  4 |       | 7  8 |
    //
    // Dense: A*B = | 1*5+2*7  1*6+2*8 | = | 19  22 |
    //              | 3*5+4*7  3*6+4*8 |   | 43  50 |
    std::vector<SparseEntry<double, 2>> a_entries = {
        {{0, 0}, 1.0}, {{0, 1}, 2.0},
        {{1, 0}, 3.0}, {{1, 1}, 4.0}};
    std::vector<SparseEntry<double, 2>> b_entries = {
        {{0, 0}, 5.0}, {{0, 1}, 6.0},
        {{1, 0}, 7.0}, {{1, 1}, 8.0}};
    CSRMatrix<double, 2, 2> A(a_entries);
    CSRMatrix<double, 2, 2> B(b_entries);

    auto C = sparse::sparse_multiply(A, B);

    CHECK(C(0, 0) == 19.0);
    CHECK(C(0, 1) == 22.0);
    CHECK(C(1, 0) == 43.0);
    CHECK(C(1, 1) == 50.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  sparse_multiply — COO * COO
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sparse_multiply coo inputs", "[sparse][multiply]") {
    COOMatrix<double, 2, 2> A;
    A.emplace(0, 0) = 1.0;
    A.emplace(0, 1) = 2.0;
    A.emplace(1, 0) = 3.0;
    A.emplace(1, 1) = 4.0;
    A.compress();

    COOMatrix<double, 2, 2> B;
    B.emplace(0, 0) = 5.0;
    B.emplace(0, 1) = 6.0;
    B.emplace(1, 0) = 7.0;
    B.emplace(1, 1) = 8.0;
    B.compress();

    auto C = sparse::sparse_multiply(A, B);
    // Returns CSRMatrix
    CHECK(C(0, 0) == 19.0);
    CHECK(C(0, 1) == 22.0);
    CHECK(C(1, 0) == 43.0);
    CHECK(C(1, 1) == 50.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  sparse_AtA
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sparse_AtA identity", "[sparse][multiply]") {
    std::vector<SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{1, 1}, 1.0}, {{2, 2}, 1.0}};
    CSRMatrix<double, 3, 3> I(entries);

    auto AtA = sparse::sparse_AtA(I);
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(AtA(i, j) == expected);
        }
    }
}

TEST_CASE("sparse_AtA general", "[sparse][multiply]") {
    // A = | 1  0 |
    //     | 0  1 |
    //     | 1  1 |
    //
    // A^T * A = | 1+0+1  0+0+1 | = | 2  1 |
    //           | 0+0+1  0+1+1 |   | 1  2 |
    std::vector<SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{1, 1}, 1.0},
        {{2, 0}, 1.0}, {{2, 1}, 1.0}};
    CSRMatrix<double, 3, 2> A(entries);

    auto AtA = sparse::sparse_AtA(A);
    static_assert(std::is_same_v<decltype(AtA), CSRMatrix<double, 2, 2>>);

    CHECK(AtA(0, 0) == 2.0);
    CHECK(AtA(0, 1) == 1.0);
    CHECK(AtA(1, 0) == 1.0);
    CHECK(AtA(1, 1) == 2.0);
}

TEST_CASE("sparse_AtA symmetric", "[sparse][multiply]") {
    // AtA is always symmetric
    std::vector<SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{0, 1}, 2.0},
        {{1, 0}, 3.0}, {{1, 1}, 4.0}};
    CSRMatrix<double, 2, 2> A(entries);

    auto AtA = sparse::sparse_AtA(A);
    CHECK(AtA(0, 1) == Catch::Approx(AtA(1, 0)).margin(1e-14));
}

// ═══════════════════════════════════════════════════════════════════════════
//  sparse_AAt
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sparse_AAt general", "[sparse][multiply]") {
    // A = | 1  0 |
    //     | 0  1 |
    //     | 1  1 |
    //
    // A * A^T = | 1+0  0+0  1+0 |   | 1  0  1 |
    //           | 0+0  0+1  0+1 | = | 0  1  1 |
    //           | 1+0  0+1  1+1 |   | 1  1  2 |
    std::vector<SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{1, 1}, 1.0},
        {{2, 0}, 1.0}, {{2, 1}, 1.0}};
    CSRMatrix<double, 3, 2> A(entries);

    auto AAt = sparse::sparse_AAt(A);
    static_assert(std::is_same_v<decltype(AAt), CSRMatrix<double, 3, 3>>);

    CHECK(AAt(0, 0) == 1.0);
    CHECK(AAt(0, 1) == 0.0);
    CHECK(AAt(0, 2) == 1.0);
    CHECK(AAt(1, 0) == 0.0);
    CHECK(AAt(1, 1) == 1.0);
    CHECK(AAt(1, 2) == 1.0);
    CHECK(AAt(2, 0) == 1.0);
    CHECK(AAt(2, 1) == 1.0);
    CHECK(AAt(2, 2) == 2.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  sparse_multiply — SpMV consistency
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sparse_multiply matches SpMV for A*B*x", "[sparse][multiply]") {
    // Verify (A*B)*x == A*(B*x) for dense x
    std::vector<SparseEntry<double, 2>> a_entries = {
        {{0, 0}, 1.0}, {{0, 2}, 2.0},
        {{1, 1}, 3.0},
        {{2, 0}, 4.0}, {{2, 2}, 5.0}};
    std::vector<SparseEntry<double, 2>> b_entries = {
        {{0, 0}, 2.0}, {{0, 1}, 1.0},
        {{1, 0}, 0.0}, {{1, 1}, 3.0},
        {{2, 0}, 1.0}, {{2, 1}, 0.0}};
    CSRMatrix<double, 3, 3> A(a_entries);
    CSRMatrix<double, 3, 2> B(b_entries);

    Vector<double, 2> x({1.0, 2.0});

    // (A*B)*x
    auto AB = sparse::sparse_multiply(A, B);
    Vector<double, 3> y1(AB * x);

    // A*(B*x)
    Vector<double, 3> Bx(B * x);
    Vector<double, 3> y2(A * Bx);

    for (index_type i = 0; i < 3; ++i) {
        CHECK(y1(i) == Catch::Approx(y2(i)).margin(1e-12));
    }
}
