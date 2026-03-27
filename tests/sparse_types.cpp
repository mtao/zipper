
#include <vector>

#include <zipper/COOMatrix.hpp>
#include <zipper/COOVector.hpp>
#include <zipper/CSMatrix.hpp>
#include <zipper/CSRMatrix.hpp>
#include <zipper/CSRVector.hpp>
#include <zipper/CSVector.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/MatrixBase.hxx>
#include <zipper/Vector.hpp>
#include <zipper/VectorBase.hxx>
#include <zipper/detail/LayoutPreference.hpp>
#include <zipper/expression/detail/ExpressionTraits.hpp>

#include "catch_include.hpp"

// ═══════════════════════════════════════════════════════════════════════════
//  SparseEntry
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sparse_entry_construction", "[sparse][entry]") {
    zipper::SparseEntry<double, 2> e{.indices = {1, 2}, .value = 3.14};
    CHECK(e.indices[0] == 1);
    CHECK(e.indices[1] == 2);
    CHECK(e.value == 3.14);

    zipper::SparseEntry<double, 1> v{.indices = {5}, .value = 2.0};
    CHECK(v.indices[0] == 5);
    CHECK(v.value == 2.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  COOMatrix — static extents
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("coo_matrix_static_default", "[sparse][coo][matrix]") {
    zipper::COOMatrix<double, 3, 3> A;
    CHECK(A.extent(0) == 3);
    CHECK(A.extent(1) == 3);
    // All entries should be zero
    CHECK(A(0, 0) == 0.0);
    CHECK(A(1, 2) == 0.0);
}

TEST_CASE("coo_matrix_static_emplace", "[sparse][coo][matrix]") {
    zipper::COOMatrix<double, 3, 4> A;
    A.emplace(0, 0) = 1.0;
    A.emplace(1, 2) = 5.0;
    A.emplace(2, 3) = 9.0;
    A.compress();

    CHECK(A(0, 0) == 1.0);
    CHECK(A(1, 2) == 5.0);
    CHECK(A(2, 3) == 9.0);
    CHECK(A(0, 1) == 0.0); // missing entry
    CHECK(A(2, 0) == 0.0);
}

TEST_CASE("coo_matrix_static_from_entries", "[sparse][coo][matrix]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{1, 1}, 2.0}, {{2, 2}, 3.0}};
    zipper::COOMatrix<double, 3, 3> A(entries);

    CHECK(A(0, 0) == 1.0);
    CHECK(A(1, 1) == 2.0);
    CHECK(A(2, 2) == 3.0);
    CHECK(A(0, 1) == 0.0);
    CHECK(A.is_compressed());
}

TEST_CASE("coo_matrix_coeff_ref", "[sparse][coo][matrix]") {
    zipper::COOMatrix<double, 3, 3> A;
    A.emplace(0, 0) = 1.0;
    A.emplace(1, 1) = 2.0;
    A.compress();

    A.coeff_ref(0, 0) = 10.0;
    CHECK(A(0, 0) == 10.0);

    CHECK_THROWS(A.coeff_ref(2, 2)); // missing entry
}

// ═══════════════════════════════════════════════════════════════════════════
//  COOMatrix — dynamic extents
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("coo_matrix_dynamic", "[sparse][coo][matrix]") {
    zipper::COOMatrix<double, zipper::dynamic_extent, zipper::dynamic_extent> A(
        4, 5);
    CHECK(A.extent(0) == 4);
    CHECK(A.extent(1) == 5);

    A.emplace(0, 0) = 1.0;
    A.emplace(3, 4) = 7.0;
    A.compress();

    CHECK(A(0, 0) == 1.0);
    CHECK(A(3, 4) == 7.0);
    CHECK(A(2, 2) == 0.0);
}

TEST_CASE("coo_matrix_dynamic_from_entries", "[sparse][coo][matrix]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {{{0, 1}, 3.0},
                                                           {{2, 0}, 4.0}};
    constexpr auto dyn = zipper::dynamic_extent;
    zipper::COOMatrix<double, dyn, dyn> A(entries,
                                          zipper::extents<dyn, dyn>(3, 3));

    CHECK(A(0, 1) == 3.0);
    CHECK(A(2, 0) == 4.0);
    CHECK(A(0, 0) == 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  COOVector
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("coo_vector_static", "[sparse][coo][vector]") {
    zipper::COOVector<double, 5> v;
    CHECK(v.extent(0) == 5);
    v.emplace(0) = 1.0;
    v.emplace(3) = 4.0;
    v.compress();

    CHECK(v(0) == 1.0);
    CHECK(v(3) == 4.0);
    CHECK(v(1) == 0.0);
    CHECK(v(4) == 0.0);
}

TEST_CASE("coo_vector_from_entries", "[sparse][coo][vector]") {
    std::vector<zipper::SparseEntry<double, 1>> entries = {{{2}, 3.14},
                                                           {{4}, 2.72}};
    zipper::COOVector<double, 10> v(entries);

    CHECK(v(2) == 3.14);
    CHECK(v(4) == 2.72);
    CHECK(v(0) == 0.0);
}

TEST_CASE("coo_vector_dynamic", "[sparse][coo][vector]") {
    zipper::COOVector<double, std::dynamic_extent> v(100);
    CHECK(v.extent(0) == 100);

    v.emplace(50) = 42.0;
    v.compress();
    CHECK(v(50) == 42.0);
    CHECK(v(0) == 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  CSRMatrix — construction and access
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("csr_matrix_from_coo", "[sparse][csr][matrix]") {
    zipper::COOMatrix<double, 3, 3> coo;
    coo.emplace(0, 0) = 1.0;
    coo.emplace(0, 2) = 2.0;
    coo.emplace(1, 1) = 3.0;
    coo.emplace(2, 0) = 4.0;
    coo.emplace(2, 2) = 5.0;
    coo.compress();

    zipper::CSRMatrix<double, 3, 3> csr(coo);

    CHECK(csr(0, 0) == 1.0);
    CHECK(csr(0, 2) == 2.0);
    CHECK(csr(1, 1) == 3.0);
    CHECK(csr(2, 0) == 4.0);
    CHECK(csr(2, 2) == 5.0);
    CHECK(csr(0, 1) == 0.0);
    CHECK(csr(1, 0) == 0.0);
}

TEST_CASE("csr_matrix_from_entries", "[sparse][csr][matrix]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{1, 1}, 2.0}, {{2, 2}, 3.0}};
    zipper::CSRMatrix<double, 3, 3> csr(entries);

    CHECK(csr(0, 0) == 1.0);
    CHECK(csr(1, 1) == 2.0);
    CHECK(csr(2, 2) == 3.0);
    CHECK(csr(0, 1) == 0.0);
}

TEST_CASE("csr_matrix_to_csr_via_method", "[sparse][csr][matrix]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {{{0, 1}, 7.0},
                                                           {{1, 0}, 8.0}};
    zipper::COOMatrix<double, 3, 3> coo(entries);

    auto csr = coo.to_csr();
    CHECK(csr(0, 1) == 7.0);
    CHECK(csr(1, 0) == 8.0);
    CHECK(csr(0, 0) == 0.0);
}

TEST_CASE("csr_matrix_coeff_ref", "[sparse][csr][matrix]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {{{0, 0}, 1.0},
                                                           {{1, 1}, 2.0}};
    zipper::CSRMatrix<double, 3, 3> csr(entries);

    csr.coeff_ref(0, 0) = 10.0;
    CHECK(csr(0, 0) == 10.0);

    CHECK_THROWS(csr.coeff_ref(2, 2)); // missing entry
}

// ═══════════════════════════════════════════════════════════════════════════
//  CSRVector
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("csr_vector_from_coo", "[sparse][csr][vector]") {
    zipper::COOVector<double, 5> coo;
    coo.emplace(1) = 3.0;
    coo.emplace(3) = 7.0;
    coo.compress();

    zipper::CSRVector<double, 5> csr(coo);
    CHECK(csr(1) == 3.0);
    CHECK(csr(3) == 7.0);
    CHECK(csr(0) == 0.0);
    CHECK(csr(2) == 0.0);
}

TEST_CASE("csr_vector_from_entries", "[sparse][csr][vector]") {
    std::vector<zipper::SparseEntry<double, 1>> entries = {{{0}, 1.0},
                                                           {{4}, 5.0}};
    zipper::CSRVector<double, 10> csr(entries);

    CHECK(csr(0) == 1.0);
    CHECK(csr(4) == 5.0);
    CHECK(csr(2) == 0.0);
}

TEST_CASE("csr_vector_to_csr_via_method", "[sparse][csr][vector]") {
    std::vector<zipper::SparseEntry<double, 1>> entries = {{{2}, 3.0},
                                                           {{7}, 9.0}};
    zipper::COOVector<double, 10> coo(entries);

    auto csr = coo.to_csr();
    CHECK(csr(2) == 3.0);
    CHECK(csr(7) == 9.0);
    CHECK(csr(0) == 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Roundtrip: COO → CSR → COO
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("roundtrip_matrix_coo_csr_coo", "[sparse][roundtrip][matrix]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{0, 2}, 2.0}, {{1, 1}, 3.0}, {{2, 0}, 4.0}};
    zipper::COOMatrix<double, 3, 3> coo(entries);
    auto csr = coo.to_csr();
    auto coo2 = csr.to_coo();

    for (const auto &e : entries) {
        CHECK(coo2(e.indices[0], e.indices[1]) == e.value);
    }
    // Check zeros preserved
    CHECK(coo2(0, 1) == 0.0);
    CHECK(coo2(1, 0) == 0.0);
    CHECK(coo2(2, 2) == 0.0);
}

TEST_CASE("roundtrip_vector_coo_csr_coo", "[sparse][roundtrip][vector]") {
    std::vector<zipper::SparseEntry<double, 1>> entries = {
        {{1}, 10.0}, {{3}, 30.0}, {{5}, 50.0}};
    zipper::COOVector<double, 10> coo(entries);
    auto csr = coo.to_csr();
    auto coo2 = csr.to_coo();

    for (const auto &e : entries) { CHECK(coo2(e.indices[0]) == e.value); }
    CHECK(coo2(0) == 0.0);
    CHECK(coo2(2) == 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  SpMV: sparse matrix * dense vector
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("coo_matrix_times_dense_vector", "[sparse][coo][spmv]") {
    // | 1  0  2 |   | 1 |   | 1+0+6 |   | 7  |
    // | 0  3  0 | * | 2 | = | 0+6+0 | = | 6  |
    // | 4  0  5 |   | 3 |   | 4+0+15|   | 19 |
    zipper::COOMatrix<double, 3, 3> A;
    A.emplace(0, 0) = 1.0;
    A.emplace(0, 2) = 2.0;
    A.emplace(1, 1) = 3.0;
    A.emplace(2, 0) = 4.0;
    A.emplace(2, 2) = 5.0;
    A.compress();

    zipper::Vector<double, 3> x({1.0, 2.0, 3.0});

    zipper::Vector<double, 3> y(A * x);
    CHECK(y(0) == 7.0);
    CHECK(y(1) == 6.0);
    CHECK(y(2) == 19.0);
}

TEST_CASE("csr_matrix_times_dense_vector", "[sparse][csr][spmv]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {{{0, 0}, 1.0},
                                                           {{0, 2}, 2.0},
                                                           {{1, 1}, 3.0},
                                                           {{2, 0}, 4.0},
                                                           {{2, 2}, 5.0}};
    zipper::CSRMatrix<double, 3, 3> A(entries);
    zipper::Vector<double, 3> x({1.0, 2.0, 3.0});

    zipper::Vector<double, 3> y(A * x);
    CHECK(y(0) == 7.0);
    CHECK(y(1) == 6.0);
    CHECK(y(2) == 19.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Concept checks
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sparse_type_concepts", "[sparse][concepts]") {
    using namespace zipper;
    static_assert(concepts::Matrix<COOMatrix<double, 3, 3>>);
    static_assert(concepts::Matrix<CSRMatrix<double, 3, 3>>);
    static_assert(concepts::Vector<COOVector<double, 5>>);
    static_assert(concepts::Vector<CSRVector<double, 5>>);

    // Dynamic extents
    constexpr auto dyn = dynamic_extent;
    static_assert(concepts::Matrix<COOMatrix<double, dyn, dyn>>);
    static_assert(concepts::Matrix<CSRMatrix<double, dyn, dyn>>);
    static_assert(concepts::Vector<COOVector<double, dyn>>);
    static_assert(concepts::Vector<CSRVector<double, dyn>>);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Copy / move semantics
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("coo_matrix_copy", "[sparse][coo][matrix]") {
    zipper::COOMatrix<double, 3, 3> A;
    A.emplace(0, 0) = 1.0;
    A.emplace(1, 1) = 2.0;
    A.compress();

    auto B = A;
    CHECK(B(0, 0) == 1.0);
    CHECK(B(1, 1) == 2.0);
}

TEST_CASE("csr_matrix_copy", "[sparse][csr][matrix]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {{{0, 0}, 1.0},
                                                           {{1, 1}, 2.0}};
    zipper::CSRMatrix<double, 3, 3> A(entries);

    auto B = A;
    CHECK(B(0, 0) == 1.0);
    CHECK(B(1, 1) == 2.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  CTAD
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("coo_matrix_ctad", "[sparse][coo][matrix]") {
    zipper::storage::SparseCoordinateAccessor<double, zipper::extents<3, 3>>
        acc;
    acc.emplace(0, 0) = 1.0;
    acc.compress();

    zipper::COOMatrix coo(std::move(acc));
    CHECK(coo(0, 0) == 1.0);
}

TEST_CASE("csr_matrix_ctad_from_coo", "[sparse][csr][matrix]") {
    zipper::COOMatrix<double, 3, 3> coo;
    coo.emplace(0, 0) = 5.0;
    coo.compress();

    zipper::CSRMatrix csr(coo);
    CHECK(csr(0, 0) == 5.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Duplicate entry handling (COO compress merges duplicates)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("coo_matrix_duplicate_entries", "[sparse][coo][matrix]") {
    zipper::COOMatrix<double, 3, 3> A;
    A.emplace(1, 1) = 3.0;
    A.emplace(1, 1) = 7.0; // duplicate — should sum
    A.compress();

    CHECK(A(1, 1) == 10.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Sparse Assignment: dense → COO
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("coo_vector_assign_from_dense", "[sparse][coo][assign]") {
    zipper::Vector<double, 5> v({1.0, 0.0, 3.0, 0.0, 5.0});

    zipper::COOVector<double, 5> sv;
    sv = v;

    CHECK(sv(0) == 1.0);
    CHECK(sv(1) == 0.0);
    CHECK(sv(2) == 3.0);
    CHECK(sv(3) == 0.0);
    CHECK(sv(4) == 5.0);
}

TEST_CASE("coo_matrix_assign_from_dense", "[sparse][coo][assign]") {
    zipper::Matrix<double, 2, 3> M;
    M(0, 0) = 1.0;
    M(0, 1) = 0.0;
    M(0, 2) = 2.0;
    M(1, 0) = 0.0;
    M(1, 1) = 3.0;
    M(1, 2) = 0.0;

    zipper::COOMatrix<double, 2, 3> A;
    A = M;

    CHECK(A(0, 0) == 1.0);
    CHECK(A(0, 1) == 0.0);
    CHECK(A(0, 2) == 2.0);
    CHECK(A(1, 0) == 0.0);
    CHECK(A(1, 1) == 3.0);
    CHECK(A(1, 2) == 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Sparse Assignment: dense → CSR
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("csr_vector_assign_from_dense", "[sparse][csr][assign]") {
    zipper::Vector<double, 5> v({0.0, 2.0, 0.0, 4.0, 0.0});

    zipper::CSRVector<double, 5> sv;
    sv = v;

    CHECK(sv(0) == 0.0);
    CHECK(sv(1) == 2.0);
    CHECK(sv(2) == 0.0);
    CHECK(sv(3) == 4.0);
    CHECK(sv(4) == 0.0);
}

TEST_CASE("csr_matrix_assign_from_dense", "[sparse][csr][assign]") {
    zipper::Matrix<double, 3, 3> M;
    M(0, 0) = 1.0;
    M(0, 1) = 0.0;
    M(0, 2) = 0.0;
    M(1, 0) = 0.0;
    M(1, 1) = 2.0;
    M(1, 2) = 0.0;
    M(2, 0) = 0.0;
    M(2, 1) = 0.0;
    M(2, 2) = 3.0;

    zipper::CSRMatrix<double, 3, 3> A;
    A = M;

    CHECK(A(0, 0) == 1.0);
    CHECK(A(1, 1) == 2.0);
    CHECK(A(2, 2) == 3.0);
    CHECK(A(0, 1) == 0.0);
    CHECK(A(1, 0) == 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Sparse Assignment: sparse → sparse (COO → COO, CSR → COO, etc.)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("coo_vector_assign_from_coo", "[sparse][coo][assign]") {
    zipper::COOVector<double, 5> src;
    src.emplace(1) = 10.0;
    src.emplace(3) = 30.0;
    src.compress();

    zipper::COOVector<double, 5> dst;
    dst = src;

    CHECK(dst(0) == 0.0);
    CHECK(dst(1) == 10.0);
    CHECK(dst(2) == 0.0);
    CHECK(dst(3) == 30.0);
    CHECK(dst(4) == 0.0);
}

TEST_CASE("coo_matrix_assign_from_csr", "[sparse][assign]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{1, 1}, 2.0}, {{2, 2}, 3.0}};
    zipper::CSRMatrix<double, 3, 3> csr(entries);

    zipper::COOMatrix<double, 3, 3> coo;
    coo = csr;

    CHECK(coo(0, 0) == 1.0);
    CHECK(coo(1, 1) == 2.0);
    CHECK(coo(2, 2) == 3.0);
    CHECK(coo(0, 1) == 0.0);
}

TEST_CASE("csr_matrix_assign_from_coo", "[sparse][assign]") {
    zipper::COOMatrix<double, 3, 3> coo;
    coo.emplace(0, 2) = 5.0;
    coo.emplace(2, 0) = 7.0;
    coo.compress();

    zipper::CSRMatrix<double, 3, 3> csr;
    csr = coo;

    CHECK(csr(0, 2) == 5.0);
    CHECK(csr(2, 0) == 7.0);
    CHECK(csr(0, 0) == 0.0);
    CHECK(csr(1, 1) == 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Sparse Assignment: expression → sparse (e.g., A * x → COO)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("coo_vector_assign_from_spmv_expr", "[sparse][coo][assign]") {
    // A = | 1  0 |,  x = | 2 |  =>  y = | 2 |
    //     | 0  3 |       | 4 |           | 12 |
    zipper::COOMatrix<double, 2, 2> A;
    A.emplace(0, 0) = 1.0;
    A.emplace(1, 1) = 3.0;
    A.compress();

    zipper::Vector<double, 2> x({2.0, 4.0});

    zipper::COOVector<double, 2> y;
    y = A * x;

    CHECK(y(0) == 2.0);
    CHECK(y(1) == 12.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Sparse Assignment: construct-from-expression
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("coo_vector_construct_from_dense", "[sparse][coo][assign]") {
    zipper::Vector<double, 3> v({1.0, 0.0, 3.0});

    zipper::COOVector<double, 3> sv(v);

    CHECK(sv(0) == 1.0);
    CHECK(sv(1) == 0.0);
    CHECK(sv(2) == 3.0);
}

TEST_CASE("coo_matrix_construct_from_dense", "[sparse][coo][assign]") {
    zipper::Matrix<double, 2, 2> M;
    M(0, 0) = 5.0;
    M(0, 1) = 0.0;
    M(1, 0) = 0.0;
    M(1, 1) = 7.0;

    zipper::COOMatrix<double, 2, 2> A(M);

    CHECK(A(0, 0) == 5.0);
    CHECK(A(0, 1) == 0.0);
    CHECK(A(1, 0) == 0.0);
    CHECK(A(1, 1) == 7.0);
}

TEST_CASE("csr_vector_construct_from_dense", "[sparse][csr][assign]") {
    zipper::Vector<double, 4> v({0.0, 2.0, 0.0, 4.0});

    zipper::CSRVector<double, 4> sv(v);

    CHECK(sv(0) == 0.0);
    CHECK(sv(1) == 2.0);
    CHECK(sv(2) == 0.0);
    CHECK(sv(3) == 4.0);
}

TEST_CASE("csr_matrix_construct_from_dense", "[sparse][csr][assign]") {
    zipper::Matrix<double, 2, 2> M;
    M(0, 0) = 1.0;
    M(0, 1) = 0.0;
    M(1, 0) = 0.0;
    M(1, 1) = 2.0;

    zipper::CSRMatrix<double, 2, 2> A(M);

    CHECK(A(0, 0) == 1.0);
    CHECK(A(0, 1) == 0.0);
    CHECK(A(1, 0) == 0.0);
    CHECK(A(1, 1) == 2.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Sparse Assignment: reassignment replaces old data
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("coo_vector_reassignment_replaces_old", "[sparse][coo][assign]") {
    zipper::COOVector<double, 3> sv;
    sv.emplace(0) = 100.0;
    sv.emplace(1) = 200.0;
    sv.emplace(2) = 300.0;
    sv.compress();

    zipper::Vector<double, 3> v({0.0, 5.0, 0.0});
    sv = v;

    // Old data should be gone; only index 1 should be nonzero.
    CHECK(sv(0) == 0.0);
    CHECK(sv(1) == 5.0);
    CHECK(sv(2) == 0.0);
}

TEST_CASE("coo_matrix_reassignment_replaces_old", "[sparse][coo][assign]") {
    zipper::COOMatrix<double, 2, 2> A;
    A.emplace(0, 0) = 10.0;
    A.emplace(0, 1) = 20.0;
    A.emplace(1, 0) = 30.0;
    A.emplace(1, 1) = 40.0;
    A.compress();

    zipper::Matrix<double, 2, 2> M;
    M(0, 0) = 0.0;
    M(0, 1) = 1.0;
    M(1, 0) = 0.0;
    M(1, 1) = 0.0;

    A = M;

    CHECK(A(0, 0) == 0.0);
    CHECK(A(0, 1) == 1.0);
    CHECK(A(1, 0) == 0.0);
    CHECK(A(1, 1) == 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  CSMatrix — unified compressed sparse matrix with layout parameter
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("csmatrix_csr_from_coo", "[sparse][csmatrix]") {
    zipper::COOMatrix<double, 3, 3> coo;
    coo.emplace(0, 0) = 1.0;
    coo.emplace(0, 2) = 2.0;
    coo.emplace(1, 1) = 3.0;
    coo.emplace(2, 0) = 4.0;
    coo.emplace(2, 2) = 5.0;
    coo.compress();

    // CSMatrix with default layout = CSR
    zipper::CSMatrix<double, 3, 3> A(coo);

    CHECK(A(0, 0) == 1.0);
    CHECK(A(0, 2) == 2.0);
    CHECK(A(1, 1) == 3.0);
    CHECK(A(2, 0) == 4.0);
    CHECK(A(2, 2) == 5.0);
    CHECK(A(0, 1) == 0.0);
    CHECK(A(1, 0) == 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  CSC (CSMatrix with layout_left) — construction and access
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("csc_matrix_from_coo", "[sparse][csc][matrix]") {
    zipper::COOMatrix<double, 3, 3> coo;
    coo.emplace(0, 0) = 1.0;
    coo.emplace(0, 2) = 2.0;
    coo.emplace(1, 1) = 3.0;
    coo.emplace(2, 0) = 4.0;
    coo.emplace(2, 2) = 5.0;
    coo.compress();

    zipper::CSMatrix<double, 3, 3, zipper::storage::layout_left> csc(coo);

    // Access should use (row, col) as normal
    CHECK(csc(0, 0) == 1.0);
    CHECK(csc(0, 2) == 2.0);
    CHECK(csc(1, 1) == 3.0);
    CHECK(csc(2, 0) == 4.0);
    CHECK(csc(2, 2) == 5.0);
    CHECK(csc(0, 1) == 0.0);
    CHECK(csc(1, 0) == 0.0);
}

TEST_CASE("csc_matrix_from_entries", "[sparse][csc][matrix]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{1, 1}, 2.0}, {{2, 2}, 3.0}};
    zipper::CSMatrix<double, 3, 3, zipper::storage::layout_left> csc(entries);

    CHECK(csc(0, 0) == 1.0);
    CHECK(csc(1, 1) == 2.0);
    CHECK(csc(2, 2) == 3.0);
    CHECK(csc(0, 1) == 0.0);
}

TEST_CASE("csc_matrix_to_csc_via_method", "[sparse][csc][matrix]") {
    zipper::COOMatrix<double, 3, 3> coo;
    coo.emplace(0, 1) = 7.0;
    coo.emplace(1, 0) = 8.0;
    coo.compress();

    auto csc = coo.to_csc();
    CHECK(csc(0, 1) == 7.0);
    CHECK(csc(1, 0) == 8.0);
    CHECK(csc(0, 0) == 0.0);
}

TEST_CASE("csc_matrix_coeff_ref", "[sparse][csc][matrix]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {{{0, 0}, 1.0},
                                                           {{1, 1}, 2.0}};
    zipper::CSMatrix<double, 3, 3, zipper::storage::layout_left> csc(entries);

    csc.coeff_ref(1, 1) = 99.0;
    CHECK(csc(1, 1) == 99.0);

    CHECK_THROWS(csc.coeff_ref(2, 2)); // missing entry
}

// ═══════════════════════════════════════════════════════════════════════════
//  CSC index_set: fast for D==0 (rows for a column), slow for D==1
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("csc_matrix_index_set", "[sparse][csc][index_set]") {
    zipper::COOMatrix<double, 4, 5> coo;
    coo.emplace(0, 1) = 1.0;
    coo.emplace(0, 3) = 2.0;
    coo.emplace(1, 2) = 3.0;
    coo.emplace(2, 0) = 4.0;
    coo.emplace(2, 3) = 5.0;
    coo.compress();

    zipper::CSMatrix<double, 4, 5, zipper::storage::layout_left> csc(coo);

    SECTION("row indices for col 0 = {2} (fast path for CSC)") {
        auto rows = csc.expression().index_set<0>(0);
        REQUIRE(rows.size() == 1);
        CHECK(*rows.begin() == 2);
    }

    SECTION("row indices for col 1 = {0} (fast path for CSC)") {
        auto rows = csc.expression().index_set<0>(1);
        REQUIRE(rows.size() == 1);
        CHECK(*rows.begin() == 0);
    }

    SECTION("row indices for col 3 = {0, 2} (fast path for CSC)") {
        auto rows = csc.expression().index_set<0>(3);
        REQUIRE(rows.size() == 2);
        auto it = rows.begin();
        CHECK(*it++ == 0);
        CHECK(*it++ == 2);
    }

    SECTION("row indices for empty col 4 = {} (fast path for CSC)") {
        auto rows = csc.expression().index_set<0>(4);
        CHECK(rows.empty());
    }

    SECTION("col indices for row 0 = {1, 3} (slow path for CSC)") {
        auto cols = csc.expression().index_set<1>(0);
        REQUIRE(cols.size() == 2);
        auto it = cols.begin();
        CHECK(*it++ == 1);
        CHECK(*it++ == 3);
    }

    SECTION("col indices for empty row 3 = {} (slow path for CSC)") {
        auto cols = csc.expression().index_set<1>(3);
        CHECK(cols.empty());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Roundtrip: COO → CSC → COO
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("roundtrip_matrix_coo_csc_coo", "[sparse][csc][roundtrip]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{0, 2}, 2.0}, {{1, 1}, 3.0}, {{2, 0}, 4.0}};
    zipper::COOMatrix<double, 3, 3> coo(entries);
    auto csc = coo.to_csc();
    auto coo2 = csc.to_coo();

    for (const auto &e : entries) {
        CHECK(coo2(e.indices[0], e.indices[1]) == e.value);
    }
    CHECK(coo2(0, 1) == 0.0);
    CHECK(coo2(1, 0) == 0.0);
    CHECK(coo2(2, 2) == 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Layout conversion: as_csr() / as_csc()
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("csmatrix_as_csr_as_csc", "[sparse][csmatrix][conversion]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {{{0, 0}, 1.0},
                                                           {{0, 2}, 2.0},
                                                           {{1, 1}, 3.0},
                                                           {{2, 0}, 4.0},
                                                           {{2, 2}, 5.0}};

    // Start with CSR
    zipper::CSMatrix<double, 3, 3, zipper::storage::layout_right> csr(entries);

    SECTION("as_csc converts CSR to CSC") {
        auto csc = csr.as_csc();
        static_assert(std::is_same_v<decltype(csc)::layout_policy,
                                     zipper::storage::layout_left>);
        CHECK(csc(0, 0) == 1.0);
        CHECK(csc(0, 2) == 2.0);
        CHECK(csc(1, 1) == 3.0);
        CHECK(csc(2, 0) == 4.0);
        CHECK(csc(2, 2) == 5.0);
        CHECK(csc(0, 1) == 0.0);
    }

    SECTION("as_csr on CSR is identity") {
        auto csr2 = csr.as_csr();
        static_assert(std::is_same_v<decltype(csr2)::layout_policy,
                                     zipper::storage::layout_right>);
        CHECK(csr2(0, 0) == 1.0);
        CHECK(csr2(2, 2) == 5.0);
    }

    SECTION("CSC → as_csr roundtrip") {
        auto csc = csr.as_csc();
        auto csr_back = csc.as_csr();
        static_assert(std::is_same_v<decltype(csr_back)::layout_policy,
                                     zipper::storage::layout_right>);
        for (const auto &e : entries) {
            CHECK(csr_back(e.indices[0], e.indices[1]) == e.value);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  CSC SpMV: sparse CSC matrix * dense vector
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("csc_matrix_times_dense_vector", "[sparse][csc][spmv]") {
    // | 1  0  2 |   | 1 |   | 7  |
    // | 0  3  0 | * | 2 | = | 6  |
    // | 4  0  5 |   | 3 |   | 19 |
    std::vector<zipper::SparseEntry<double, 2>> entries = {{{0, 0}, 1.0},
                                                           {{0, 2}, 2.0},
                                                           {{1, 1}, 3.0},
                                                           {{2, 0}, 4.0},
                                                           {{2, 2}, 5.0}};
    zipper::CSMatrix<double, 3, 3, zipper::storage::layout_left> csc(entries);
    zipper::Vector<double, 3> x({1.0, 2.0, 3.0});

    zipper::Vector<double, 3> y(csc * x);
    CHECK(y(0) == 7.0);
    CHECK(y(1) == 6.0);
    CHECK(y(2) == 19.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  CSVector (unified sparse vector, no layout param)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("csvector_from_coo", "[sparse][csvector]") {
    zipper::COOVector<double, 5> coo;
    coo.emplace(1) = 3.0;
    coo.emplace(3) = 7.0;
    coo.compress();

    zipper::CSVector<double, 5> sv(coo);
    CHECK(sv(1) == 3.0);
    CHECK(sv(3) == 7.0);
    CHECK(sv(0) == 0.0);
}

TEST_CASE("csvector_from_entries", "[sparse][csvector]") {
    std::vector<zipper::SparseEntry<double, 1>> entries = {{{0}, 1.0},
                                                           {{4}, 5.0}};
    zipper::CSVector<double, 10> sv(entries);

    CHECK(sv(0) == 1.0);
    CHECK(sv(4) == 5.0);
    CHECK(sv(2) == 0.0);
}

TEST_CASE("csvector_construct_from_dense", "[sparse][csvector]") {
    zipper::Vector<double, 4> v({0.0, 2.0, 0.0, 4.0});

    zipper::CSVector<double, 4> sv(v);

    CHECK(sv(0) == 0.0);
    CHECK(sv(1) == 2.0);
    CHECK(sv(2) == 0.0);
    CHECK(sv(3) == 4.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Concept checks for new types
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("csmatrix_csvector_concepts",
          "[sparse][csmatrix][csvector][concepts]") {
    using namespace zipper;

    // CSMatrix with CSR layout
    static_assert(concepts::Matrix<CSMatrix<double, 3, 3>>);
    static_assert(
        concepts::Matrix<CSMatrix<double, 3, 3, storage::layout_right>>);

    // CSMatrix with CSC layout
    static_assert(
        concepts::Matrix<CSMatrix<double, 3, 3, storage::layout_left>>);

    // CSVector
    static_assert(concepts::Vector<CSVector<double, 5>>);

    // Dynamic extents
    constexpr auto dyn = dynamic_extent;
    static_assert(concepts::Matrix<CSMatrix<double, dyn, dyn>>);
    static_assert(
        concepts::Matrix<CSMatrix<double, dyn, dyn, storage::layout_left>>);
    static_assert(concepts::Vector<CSVector<double, dyn>>);
}

// ═══════════════════════════════════════════════════════════════════════════
//  CSC Assignment from dense
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("csc_matrix_assign_from_dense", "[sparse][csc][assign]") {
    zipper::Matrix<double, 3, 3> M;
    M(0, 0) = 1.0;
    M(0, 1) = 0.0;
    M(0, 2) = 0.0;
    M(1, 0) = 0.0;
    M(1, 1) = 2.0;
    M(1, 2) = 0.0;
    M(2, 0) = 0.0;
    M(2, 1) = 0.0;
    M(2, 2) = 3.0;

    zipper::CSMatrix<double, 3, 3, zipper::storage::layout_left> csc;
    csc = M;

    CHECK(csc(0, 0) == 1.0);
    CHECK(csc(1, 1) == 2.0);
    CHECK(csc(2, 2) == 3.0);
    CHECK(csc(0, 1) == 0.0);
    CHECK(csc(1, 0) == 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Compile-time: preferred_layout propagation
// ═══════════════════════════════════════════════════════════════════════════

namespace {
namespace lp = zipper::detail;
template <typename T>
using pref = typename zipper::expression::detail::ExpressionTraits<
    std::decay_t<T>>::preferred_layout;

// Underlying expression type of a ZipperBase wrapper
template <typename Wrapper>
using expr_of = typename Wrapper::expression_type;

// ── Leaf preferences ─────────────────────────────────────────────────────

// Dense row-major Matrix → DenseLayoutPreference<layout_right>
static_assert(lp::is_dense_layout_preference_v<
              pref<expr_of<zipper::Matrix<double, 3, 3>>>>);
static_assert(std::is_same_v<pref<expr_of<zipper::Matrix<double, 3, 3>>>,
                             lp::PreferRowMajor>);

// Dense col-major Matrix → DenseLayoutPreference<layout_left>
static_assert(std::is_same_v<pref<expr_of<zipper::Matrix<double, 3, 3, false>>>,
                             lp::PreferColMajor>);

// CSR matrix → SparseLayoutPreference<layout_right>
static_assert(lp::is_sparse_layout_preference_v<
              pref<expr_of<zipper::CSRMatrix<double, 3, 3>>>>);
static_assert(std::is_same_v<pref<expr_of<zipper::CSRMatrix<double, 3, 3>>>,
                             lp::PreferCSR>);

// CSC matrix → SparseLayoutPreference<layout_left>
static_assert(
    std::is_same_v<
        pref<expr_of<
            zipper::CSMatrix<double, 3, 3, zipper::storage::layout_left>>>,
        lp::PreferCSC>);

// COO matrix → NoLayoutPreference
static_assert(lp::is_no_layout_preference_v<
              pref<expr_of<zipper::COOMatrix<double, 3, 3>>>>);

// ── Transpose flipping ──────────────────────────────────────────────────

// CSR.transpose() → PreferCSC
using CSR33Expr = expr_of<zipper::CSRMatrix<double, 3, 3>>;
using TransposedCSR =
    zipper::expression::unary::Swizzle<const CSR33Expr &, 1, 0>;
static_assert(std::is_same_v<pref<TransposedCSR>, lp::PreferCSC>);

// CSC.transpose() → PreferCSR
using CSC33Expr =
    expr_of<zipper::CSMatrix<double, 3, 3, zipper::storage::layout_left>>;
using TransposedCSC =
    zipper::expression::unary::Swizzle<const CSC33Expr &, 1, 0>;
static_assert(std::is_same_v<pref<TransposedCSC>, lp::PreferCSR>);

// Dense row-major.transpose() → PreferColMajor
using DenseRMExpr = expr_of<zipper::Matrix<double, 3, 3>>;
using TransposedDenseRM =
    zipper::expression::unary::Swizzle<const DenseRMExpr &, 1, 0>;
static_assert(std::is_same_v<pref<TransposedDenseRM>, lp::PreferColMajor>);

// Dense col-major.transpose() → PreferRowMajor
using DenseCMExpr = expr_of<zipper::Matrix<double, 3, 3, false>>;
using TransposedDenseCM =
    zipper::expression::unary::Swizzle<const DenseCMExpr &, 1, 0>;
static_assert(std::is_same_v<pref<TransposedDenseCM>, lp::PreferRowMajor>);

} // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
//  Runtime: smart eval() dispatch
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("eval_csr_produces_csmatrix_csr", "[sparse][eval]") {
    // Build a CSR matrix
    zipper::COOMatrix<double, 3, 3> coo;
    coo.emplace(0, 0) = 1.0;
    coo.emplace(1, 1) = 2.0;
    coo.emplace(2, 2) = 3.0;
    zipper::CSRMatrix<double, 3, 3> csr(coo);

    // eval() on a CSR expression should produce a CSMatrix<..., layout_right>
    auto result = csr.transpose().transpose().eval();
    using result_type = decltype(result);
    static_assert(
        std::is_same_v<
            result_type,
            zipper::CSMatrix<double, 3, 3, zipper::storage::layout_right>>);
    CHECK(result(0, 0) == 1.0);
    CHECK(result(1, 1) == 2.0);
    CHECK(result(2, 2) == 3.0);
}

TEST_CASE("eval_csr_transpose_produces_csc", "[sparse][eval]") {
    zipper::COOMatrix<double, 2, 3> coo;
    coo.emplace(0, 0) = 1.0;
    coo.emplace(0, 2) = 2.0;
    coo.emplace(1, 1) = 3.0;
    zipper::CSMatrix<double, 2, 3, zipper::storage::layout_right> csr(coo);

    // transpose of CSR → eval() should produce CSC
    auto result = csr.transpose().eval();
    using result_type = decltype(result);
    static_assert(
        std::is_same_v<
            result_type,
            zipper::CSMatrix<double, 3, 2, zipper::storage::layout_left>>);
    // Transposed values: result(col, row) = original(row, col)
    CHECK(result(0, 0) == 1.0);
    CHECK(result(2, 0) == 2.0);
    CHECK(result(1, 1) == 3.0);
    CHECK(result(0, 1) == 0.0);
}

TEST_CASE("eval_csc_transpose_produces_csr", "[sparse][eval]") {
    zipper::COOMatrix<double, 3, 2> coo;
    coo.emplace(0, 0) = 5.0;
    coo.emplace(2, 1) = 7.0;
    zipper::CSMatrix<double, 3, 2, zipper::storage::layout_left> csc(coo);

    // transpose of CSC → eval() should produce CSR
    auto result = csc.transpose().eval();
    using result_type = decltype(result);
    static_assert(
        std::is_same_v<
            result_type,
            zipper::CSMatrix<double, 2, 3, zipper::storage::layout_right>>);
    CHECK(result(0, 0) == 5.0);
    CHECK(result(1, 2) == 7.0);
    CHECK(result(0, 1) == 0.0);
}

TEST_CASE("eval_dense_still_produces_matrix", "[sparse][eval]") {
    zipper::Matrix<double, 2, 2> A({{1.0, 2.0}, {3.0, 4.0}});

    auto result = A.transpose().transpose().eval();
    using result_type = decltype(result);
    // Dense row-major → transpose → transpose → still row-major dense
    static_assert(std::is_same_v<result_type, zipper::Matrix<double, 2, 2>>);
    CHECK(result(0, 0) == 1.0);
    CHECK(result(1, 1) == 4.0);
}

TEST_CASE("eval_dense_transpose_produces_col_major", "[sparse][eval]") {
    zipper::Matrix<double, 2, 3> A({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});

    auto result = A.transpose().eval();
    using result_type = decltype(result);
    // Dense row-major transposed → should be col-major dense
    static_assert(
        std::is_same_v<result_type, zipper::Matrix<double, 3, 2, false>>);
    CHECK(result(0, 0) == 1.0);
    CHECK(result(0, 1) == 4.0);
    CHECK(result(2, 0) == 3.0);
    CHECK(result(2, 1) == 6.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Audit fix #2: Uncompressed find() works without assertion failure
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("coo_uncompressed_find", "[sparse][coo][audit][bug2]") {
    // Before the fix, calling operator() on an uncompressed COO would hit
    // an assertion in iterator operator<=> that unconditionally required
    // m_compressed. Now find() falls back to a linear scan.
    zipper::COOMatrix<double, 3, 3> A;
    A.emplace(0, 0) = 1.0;
    A.emplace(1, 1) = 2.0;
    // Deliberately do NOT call compress()

    // These should work via the uncompressed linear-scan path
    CHECK(A(0, 0) == 1.0);
    CHECK(A(1, 1) == 2.0);
    CHECK(A(2, 2) == 0.0); // missing entry in uncompressed state
}

TEST_CASE("coo_vector_uncompressed_find",
          "[sparse][coo][vector][audit][bug2]") {
    zipper::COOVector<double, 5> v;
    v.emplace(1) = 10.0;
    v.emplace(3) = 30.0;
    // Do NOT compress

    CHECK(v(1) == 10.0);
    CHECK(v(3) == 30.0);
    CHECK(v(0) == 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Audit fix #3: coeff_ref/const_coeff_ref error message says
//  "SparseCoordinateAccessor" (not "SparseCoordinateStorage")
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("coo_coeff_ref_error_message", "[sparse][coo][audit][bug3]") {
    zipper::COOMatrix<double, 3, 3> A;
    A.emplace(0, 0) = 1.0;
    A.compress();

    try {
        A.coeff_ref(2, 2); // missing entry
        FAIL("Should have thrown");
    } catch (const std::invalid_argument &e) {
        std::string msg = e.what();
        CHECK(msg.find("SparseCoordinateAccessor") != std::string::npos);
        // Ensure old incorrect name is NOT present
        CHECK(msg.find("SparseCoordinateStorage") == std::string::npos);
    }
}

TEST_CASE("coo_const_coeff_ref_error_message", "[sparse][coo][audit][bug3]") {
    zipper::COOMatrix<double, 3, 3> A;
    A.emplace(0, 0) = 1.0;
    A.compress();

    const auto &cA = A;
    try {
        cA.const_coeff_ref(2, 2); // missing entry
        FAIL("Should have thrown");
    } catch (const std::invalid_argument &e) {
        std::string msg = e.what();
        CHECK(msg.find("SparseCoordinateAccessor") != std::string::npos);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Audit fixes #5-8: Constraint checks and element_type alias
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("coo_element_type_alias", "[sparse][coo][audit][fix8]") {
    // Fix #8: COO should have element_type = remove_const_t<ValueType>
    using COO =
        zipper::storage::SparseCoordinateAccessor<double,
                                                  zipper::extents<3, 3>>;
    static_assert(std::is_same_v<COO::element_type, double>);
    static_assert(std::is_same_v<COO::value_type, double>);

    // Also verify on compressed accessor (supports const ValueType)
    using ConstCS =
        zipper::storage::SparseCompressedAccessor<const double,
                                                  zipper::extents<3, 3>>;
    static_assert(std::is_same_v<ConstCS::element_type, double>);
    static_assert(std::is_same_v<ConstCS::value_type, const double>);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Audit fix #9: VectorBase::eval() sparse-aware dispatch
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("eval_csr_vector_produces_csvector",
          "[sparse][vector][eval][audit][fix9]") {
    zipper::COOVector<double, 5> coo;
    coo.emplace(1) = 3.0;
    coo.emplace(3) = 7.0;
    coo.compress();

    zipper::CSRVector<double, 5> csr(coo);

    // eval() on a CSR vector expression should produce CSVector, not dense
    // Vector
    auto result = csr.eval();
    using result_type = decltype(result);
    static_assert(std::is_same_v<result_type, zipper::CSVector<double, 5>>);
    CHECK(result(1) == 3.0);
    CHECK(result(3) == 7.0);
    CHECK(result(0) == 0.0);
}

TEST_CASE("eval_csvector_produces_csvector",
          "[sparse][vector][eval][audit][fix9]") {
    zipper::COOVector<double, 5> coo;
    coo.emplace(2) = 42.0;
    coo.compress();

    zipper::CSVector<double, 5> sv(coo);

    auto result = sv.eval();
    using result_type = decltype(result);
    static_assert(std::is_same_v<result_type, zipper::CSVector<double, 5>>);
    CHECK(result(2) == 42.0);
    CHECK(result(0) == 0.0);
}

TEST_CASE("eval_dense_vector_still_produces_vector",
          "[sparse][vector][eval][audit][fix9]") {
    zipper::Vector<double, 3> v({1.0, 2.0, 3.0});

    auto result = v.eval();
    using result_type = decltype(result);
    // Dense vector eval() should still produce a dense Vector
    static_assert(std::is_same_v<result_type, zipper::Vector<double, 3>>);
    CHECK(result(0) == 1.0);
    CHECK(result(2) == 3.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Audit fix #10: COOVector::to_cs() → CSVector
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("coo_vector_to_cs", "[sparse][coo][vector][audit][fix10]") {
    zipper::COOVector<double, 5> coo;
    coo.emplace(0) = 1.0;
    coo.emplace(2) = 3.0;
    coo.emplace(4) = 5.0;
    coo.compress();

    auto sv = coo.to_cs();
    using result_type = decltype(sv);
    static_assert(std::is_same_v<result_type, zipper::CSVector<double, 5>>);

    CHECK(sv(0) == 1.0);
    CHECK(sv(1) == 0.0);
    CHECK(sv(2) == 3.0);
    CHECK(sv(3) == 0.0);
    CHECK(sv(4) == 5.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Audit fix #12: CSRMatrix::layout_policy alias
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("csr_matrix_layout_policy_alias", "[sparse][csr][audit][fix12]") {
    static_assert(std::is_same_v<zipper::CSRMatrix<double, 3, 3>::layout_policy,
                                 zipper::storage::layout_right>);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Audit fix #13: SparseCompressedAccessor::clear()
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("compressed_accessor_clear", "[sparse][compressed][audit][fix13]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{1, 1}, 2.0}, {{2, 2}, 3.0}};
    zipper::CSRMatrix<double, 3, 3> csr(entries);

    CHECK(csr(0, 0) == 1.0);
    CHECK(csr(1, 1) == 2.0);
    CHECK(csr(2, 2) == 3.0);

    // clear() should remove all entries
    csr.expression().clear();

    CHECK(csr(0, 0) == 0.0);
    CHECK(csr(1, 1) == 0.0);
    CHECK(csr(2, 2) == 0.0);
}

TEST_CASE("csc_accessor_clear", "[sparse][csc][compressed][audit][fix13]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {{{0, 1}, 5.0},
                                                           {{2, 0}, 7.0}};
    zipper::CSMatrix<double, 3, 3, zipper::storage::layout_left> csc(entries);

    CHECK(csc(0, 1) == 5.0);
    CHECK(csc(2, 0) == 7.0);

    csc.expression().clear();

    CHECK(csc(0, 1) == 0.0);
    CHECK(csc(2, 0) == 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Audit fix #16: const_coeff_ref on const SparseCompressedAccessor
//  (no longer gated on !is_const_v<ValueType>)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("compressed_const_coeff_ref_on_const_accessor",
          "[sparse][compressed][audit][fix16]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {{{0, 0}, 1.0},
                                                           {{1, 1}, 2.0}};
    zipper::CSRMatrix<double, 3, 3> csr(entries);

    // const_coeff_ref should work via const reference
    const auto &expr = csr.expression();
    CHECK(expr.const_coeff_ref(0, 0) == 1.0);
    CHECK(expr.const_coeff_ref(1, 1) == 2.0);
    CHECK_THROWS(expr.const_coeff_ref(2, 2)); // missing
}

TEST_CASE("compressed_const_coeff_ref_on_const_value_type",
          "[sparse][compressed][audit][fix16]") {
    // Build a const-value-type accessor (read-only)
    using ConstCSR = zipper::storage::SparseCompressedAccessor<
        const double,
        zipper::extents<3, 3>,
        zipper::storage::layout_right>;

    // const_coeff_ref should compile even for const ValueType
    // (previously gated on !is_const_v<ValueType>)
    static_assert(requires(const ConstCSR &a) { a.const_coeff_ref(0, 0); });
}

// ═══════════════════════════════════════════════════════════════════════════
//  Audit fix #17: CSC-aware SparseAssignHelper iteration
//  Assigning CSC → COO should produce correct values
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("csc_to_coo_assign", "[sparse][csc][assign][audit][fix17]") {
    // Build a CSC matrix
    std::vector<zipper::SparseEntry<double, 2>> entries = {{{0, 0}, 1.0},
                                                           {{0, 2}, 2.0},
                                                           {{1, 1}, 3.0},
                                                           {{2, 0}, 4.0},
                                                           {{2, 2}, 5.0}};
    zipper::CSMatrix<double, 3, 3, zipper::storage::layout_left> csc(entries);

    // Assign CSC → COO
    zipper::COOMatrix<double, 3, 3> coo;
    coo = csc;

    CHECK(coo(0, 0) == 1.0);
    CHECK(coo(0, 2) == 2.0);
    CHECK(coo(1, 1) == 3.0);
    CHECK(coo(2, 0) == 4.0);
    CHECK(coo(2, 2) == 5.0);
    CHECK(coo(0, 1) == 0.0);
    CHECK(coo(1, 0) == 0.0);
}

TEST_CASE("csc_to_csr_assign", "[sparse][csc][csr][assign][audit][fix17]") {
    // CSC → CSR via assignment (goes through SparseAssignHelper)
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 0}, 10.0}, {{1, 0}, 20.0}, {{1, 2}, 30.0}};
    zipper::CSMatrix<double, 2, 3, zipper::storage::layout_left> csc(entries);

    zipper::CSRMatrix<double, 2, 3> csr;
    csr = csc;

    CHECK(csr(0, 0) == 10.0);
    CHECK(csr(1, 0) == 20.0);
    CHECK(csr(1, 2) == 30.0);
    CHECK(csr(0, 1) == 0.0);
    CHECK(csr(0, 2) == 0.0);
    CHECK(csr(1, 1) == 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Audit fix #18: Compressed-to-compressed fast path (same layout)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("compressed_to_compressed_same_layout_fast_path",
          "[sparse][compressed][assign][audit][fix18]") {
    // CSR → CSR: fast path should directly copy compressed data
    std::vector<zipper::SparseEntry<double, 2>> entries = {{{0, 0}, 1.0},
                                                           {{0, 2}, 2.0},
                                                           {{1, 1}, 3.0},
                                                           {{2, 0}, 4.0},
                                                           {{2, 2}, 5.0}};
    zipper::CSRMatrix<double, 3, 3> src(entries);

    zipper::CSRMatrix<double, 3, 3> dst;
    dst = src; // should use fast path

    CHECK(dst(0, 0) == 1.0);
    CHECK(dst(0, 2) == 2.0);
    CHECK(dst(1, 1) == 3.0);
    CHECK(dst(2, 0) == 4.0);
    CHECK(dst(2, 2) == 5.0);
    CHECK(dst(0, 1) == 0.0);
}

TEST_CASE("compressed_to_compressed_csc_fast_path",
          "[sparse][csc][compressed][assign][audit][fix18]") {
    // CSC → CSC: fast path
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 0}, 10.0}, {{1, 1}, 20.0}, {{2, 2}, 30.0}};
    zipper::CSMatrix<double, 3, 3, zipper::storage::layout_left> src(entries);

    zipper::CSMatrix<double, 3, 3, zipper::storage::layout_left> dst;
    dst = src;

    CHECK(dst(0, 0) == 10.0);
    CHECK(dst(1, 1) == 20.0);
    CHECK(dst(2, 2) == 30.0);
    CHECK(dst(0, 1) == 0.0);
}

TEST_CASE("compressed_vector_same_layout_fast_path",
          "[sparse][vector][compressed][assign][audit][fix18]") {
    // CSVector → CSVector: fast path (rank-1, layout-agnostic)
    std::vector<zipper::SparseEntry<double, 1>> entries = {{{1}, 5.0},
                                                           {{3}, 15.0}};
    zipper::CSVector<double, 5> src(entries);

    zipper::CSVector<double, 5> dst;
    dst = src;

    CHECK(dst(0) == 0.0);
    CHECK(dst(1) == 5.0);
    CHECK(dst(2) == 0.0);
    CHECK(dst(3) == 15.0);
    CHECK(dst(4) == 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Audit fix #19: COO reserve() method
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("coo_reserve", "[sparse][coo][audit][fix19]") {
    // reserve() should not change behavior, only avoid reallocations.
    // Verify it doesn't corrupt data.
    zipper::storage::SparseCoordinateAccessor<double, zipper::extents<10>> acc;
    acc.reserve(5); // pre-allocate

    acc.emplace(0) = 1.0;
    acc.emplace(3) = 3.0;
    acc.emplace(7) = 7.0;
    acc.compress();

    CHECK(acc.coeff(0) == 1.0);
    CHECK(acc.coeff(3) == 3.0);
    CHECK(acc.coeff(7) == 7.0);
    CHECK(acc.coeff(1) == 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Audit fix #10 (cont.): COOMatrix::to_cs<LayoutPolicy>() template
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("coo_matrix_to_cs_csr", "[sparse][coo][matrix][audit][fix10]") {
    zipper::COOMatrix<double, 3, 3> coo;
    coo.emplace(0, 0) = 1.0;
    coo.emplace(1, 1) = 2.0;
    coo.emplace(2, 2) = 3.0;
    coo.compress();

    auto csr = coo.to_cs<zipper::storage::layout_right>();
    using result_type = decltype(csr);
    static_assert(
        std::is_same_v<
            result_type,
            zipper::CSMatrix<double, 3, 3, zipper::storage::layout_right>>);

    CHECK(csr(0, 0) == 1.0);
    CHECK(csr(1, 1) == 2.0);
    CHECK(csr(2, 2) == 3.0);
}

TEST_CASE("coo_matrix_to_cs_csc", "[sparse][coo][matrix][audit][fix10]") {
    zipper::COOMatrix<double, 3, 3> coo;
    coo.emplace(0, 1) = 5.0;
    coo.emplace(2, 0) = 7.0;
    coo.compress();

    auto csc = coo.to_cs<zipper::storage::layout_left>();
    using result_type = decltype(csc);
    static_assert(
        std::is_same_v<
            result_type,
            zipper::CSMatrix<double, 3, 3, zipper::storage::layout_left>>);

    CHECK(csc(0, 1) == 5.0);
    CHECK(csc(2, 0) == 7.0);
    CHECK(csc(0, 0) == 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Audit fix #14: COO clear() method (on SparseCoordinateAccessor)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("coo_accessor_clear", "[sparse][coo][audit][fix14]") {
    zipper::COOMatrix<double, 3, 3> A;
    A.emplace(0, 0) = 1.0;
    A.emplace(1, 1) = 2.0;
    A.compress();

    CHECK(A(0, 0) == 1.0);
    CHECK(A(1, 1) == 2.0);

    A.expression().clear();

    CHECK(A(0, 0) == 0.0);
    CHECK(A(1, 1) == 0.0);
    CHECK(A.is_compressed()); // clear() sets compressed = true
}

// ═══════════════════════════════════════════════════════════════════════════
//  Sparse span views: CSMatrix::as_span() / as_const_span()
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("csmatrix_csr_as_span_reads", "[sparse][csmatrix][span]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {{{0, 0}, 1.0},
                                                           {{0, 2}, 2.0},
                                                           {{1, 1}, 3.0},
                                                           {{2, 0}, 4.0},
                                                           {{2, 2}, 5.0}};
    zipper::CSMatrix<double, 3, 3, zipper::storage::layout_right> csr(entries);

    auto span = csr.as_span();
    // Existing entries: mutable operator() returns reference.
    CHECK(span(0, 0) == 1.0);
    CHECK(span(0, 2) == 2.0);
    CHECK(span(1, 1) == 3.0);
    CHECK(span(2, 0) == 4.0);
    CHECK(span(2, 2) == 5.0);
    // Missing entries: use const access (coeff returns 0).
    CHECK(std::as_const(span)(0, 1) == 0.0);
}

TEST_CASE("csmatrix_csc_as_span_reads", "[sparse][csmatrix][span]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {{{0, 0}, 1.0},
                                                           {{0, 2}, 2.0},
                                                           {{1, 1}, 3.0},
                                                           {{2, 0}, 4.0},
                                                           {{2, 2}, 5.0}};
    zipper::CSMatrix<double, 3, 3, zipper::storage::layout_left> csc(entries);

    auto span = csc.as_span();
    CHECK(span(0, 0) == 1.0);
    CHECK(span(0, 2) == 2.0);
    CHECK(span(1, 1) == 3.0);
    CHECK(span(2, 0) == 4.0);
    CHECK(span(2, 2) == 5.0);
    // Missing entries: use const access (coeff returns 0).
    CHECK(std::as_const(span)(0, 1) == 0.0);
}

TEST_CASE("csmatrix_as_span_mutates", "[sparse][csmatrix][span]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{1, 1}, 2.0}, {{2, 2}, 3.0}};
    zipper::CSMatrix<double, 3, 3, zipper::storage::layout_right> csr(entries);

    // Modify through the span view.
    auto span = csr.as_span();
    span(1, 1) = 99.0;

    // The original matrix should see the change (non-owning view).
    CHECK(csr(1, 1) == 99.0);
}

TEST_CASE("csmatrix_as_const_span_reads", "[sparse][csmatrix][span]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {{{0, 0}, 1.0},
                                                           {{1, 1}, 2.0}};
    zipper::CSMatrix<double, 3, 3, zipper::storage::layout_right> csr(entries);

    auto cspan = csr.as_const_span();
    CHECK(cspan(0, 0) == 1.0);
    CHECK(cspan(1, 1) == 2.0);
    CHECK(cspan(0, 1) == 0.0);
}

TEST_CASE("csmatrix_const_as_span_gives_const", "[sparse][csmatrix][span]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {{{0, 0}, 1.0},
                                                           {{1, 1}, 2.0}};
    const zipper::CSMatrix<double, 3, 3, zipper::storage::layout_right> csr(
        entries);

    // Calling as_span() on a const CSMatrix should return const_span_type.
    auto cspan = csr.as_span();
    CHECK(cspan(0, 0) == 1.0);
    CHECK(cspan(1, 1) == 2.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Sparse span views: CSVector::as_span() / as_const_span()
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("csvector_as_span_reads", "[sparse][csvector][span]") {
    std::vector<zipper::SparseEntry<double, 1>> entries = {{{1}, 3.0},
                                                           {{3}, 7.0}};
    zipper::CSVector<double, 5> sv(entries);

    auto span = sv.as_span();
    // Existing entries via mutable operator().
    CHECK(span(1) == 3.0);
    CHECK(span(3) == 7.0);
    // Missing entries: use const access (coeff returns 0).
    CHECK(std::as_const(span)(0) == 0.0);
    CHECK(std::as_const(span)(2) == 0.0);
    CHECK(std::as_const(span)(4) == 0.0);
}

TEST_CASE("csvector_as_span_mutates", "[sparse][csvector][span]") {
    std::vector<zipper::SparseEntry<double, 1>> entries = {
        {{0}, 1.0}, {{2}, 3.0}, {{4}, 5.0}};
    zipper::CSVector<double, 5> sv(entries);

    auto span = sv.as_span();
    span(2) = 42.0;

    // Original should see the mutation.
    CHECK(sv(2) == 42.0);
}

TEST_CASE("csvector_as_const_span_reads", "[sparse][csvector][span]") {
    std::vector<zipper::SparseEntry<double, 1>> entries = {{{1}, 10.0},
                                                           {{3}, 30.0}};
    zipper::CSVector<double, 5> sv(entries);

    auto cspan = sv.as_const_span();
    CHECK(cspan(1) == 10.0);
    CHECK(cspan(3) == 30.0);
    CHECK(cspan(0) == 0.0);
}

TEST_CASE("csvector_const_as_span_gives_const", "[sparse][csvector][span]") {
    std::vector<zipper::SparseEntry<double, 1>> entries = {{{2}, 42.0}};
    const zipper::CSVector<double, 5> sv(entries);

    auto cspan = sv.as_span();
    CHECK(cspan(2) == 42.0);
    CHECK(cspan(0) == 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Sparse span views: SpMV through span view
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("sparse_span_spmv", "[sparse][csmatrix][span][spmv]") {
    // | 1  0  2 |   | 1 |   | 7  |
    // | 0  3  0 | * | 2 | = | 6  |
    // | 4  0  5 |   | 3 |   | 19 |
    std::vector<zipper::SparseEntry<double, 2>> entries = {{{0, 0}, 1.0},
                                                           {{0, 2}, 2.0},
                                                           {{1, 1}, 3.0},
                                                           {{2, 0}, 4.0},
                                                           {{2, 2}, 5.0}};
    zipper::CSMatrix<double, 3, 3, zipper::storage::layout_right> csr(entries);

    auto span = csr.as_const_span();
    zipper::Vector<double, 3> x({1.0, 2.0, 3.0});

    zipper::Vector<double, 3> y(span * x);
    CHECK(y(0) == 7.0);
    CHECK(y(1) == 6.0);
    CHECK(y(2) == 19.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Dynamic-extent converting constructors from expressions
//
//  BUG: CSRMatrix, COOMatrix, and CSMatrix have converting constructors
//  that accept any concepts::Expression or concepts::Zipper with compatible
//  extents. These constructors implicitly default-construct the Base chain
//  (CSRMatrix -> MatrixBase -> ZipperBase), then try to set extents and
//  assign in the body. But for dynamic-extent sparse types, the underlying
//  accessor (SparseCompressedAccessor / SparseCoordinateAccessor) has NO
//  default constructor — its default ctor is `requires(IsStatic)`. So the
//  Base's default ctor is implicitly deleted, and these converting
//  constructors fail to compile.
//
//  The fix: use a member initializer list to construct Base with an
//  expression_type that already has the correct extents, e.g.:
//    CSRMatrix(const Other &other)
//      : Base(expression_type(extents_traits::convert_from(other.extents())))
//    { expression().assign(other); }
//
//  These tests verify that dynamic-extent sparse types can be constructed
//  from:
//    1. Static-extent sparse matrices (static -> dynamic conversion)
//    2. Lazy transpose expressions (Zipper-wrapped Swizzle)
//    3. Lazy matrix product expressions (Zipper-wrapped MatrixProduct)
//    4. The .eval() path (which routes through MatrixBase::eval() -> CSMatrix)
// ═══════════════════════════════════════════════════════════════════════════

constexpr auto dyn = zipper::dynamic_extent;

TEST_CASE("csrmatrix_dynamic_from_static_csrmatrix",
          "[sparse][csr][dynamic][converting_ctor]") {
    // Construct a static-extent CSR, then convert to dynamic-extent CSR.
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{0, 1}, 2.0}, {{1, 0}, 3.0}, {{1, 1}, 4.0}};
    zipper::CSRMatrix<double, 2, 2> A(entries);

    // This should invoke CSRMatrix(const Zipper &other) with !is_static,
    // which currently fails because Base() is deleted for dynamic extents.
    zipper::CSRMatrix<double, dyn, dyn> B(A);

    CHECK(B.rows() == 2);
    CHECK(B.cols() == 2);
    CHECK(B(0, 0) == 1.0);
    CHECK(B(0, 1) == 2.0);
    CHECK(B(1, 0) == 3.0);
    CHECK(B(1, 1) == 4.0);
}

TEST_CASE("coomatrix_dynamic_from_static_coomatrix",
          "[sparse][coo][dynamic][converting_ctor]") {
    // Construct a static-extent COO, then convert to dynamic-extent COO.
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{1, 1}, 5.0}, {{2, 0}, 9.0}};
    zipper::COOMatrix<double, 3, 2> A(entries);

    // This should invoke COOMatrix(const Zipper &other) with !is_static.
    zipper::COOMatrix<double, dyn, dyn> B(A);

    CHECK(B.rows() == 3);
    CHECK(B.cols() == 2);
    CHECK(B(0, 0) == 1.0);
    CHECK(B(1, 1) == 5.0);
    CHECK(B(2, 0) == 9.0);
    CHECK(B(0, 1) == 0.0);
}

TEST_CASE("csrmatrix_dynamic_from_transpose_expression",
          "[sparse][csr][dynamic][converting_ctor]") {
    // Build a 2x3 static CSR, transpose it (lazy Swizzle expression),
    // and construct a dynamic-extent CSR from the transposed view.
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{0, 2}, 2.0}, {{1, 1}, 3.0}};
    zipper::CSRMatrix<double, 2, 3> A(entries);

    // A.transpose() returns MatrixBase<Swizzle<...>> — a Zipper type.
    // Constructing a dynamic CSR from it exercises the Zipper converting ctor.
    zipper::CSRMatrix<double, dyn, dyn> At(A.transpose());

    CHECK(At.rows() == 3);
    CHECK(At.cols() == 2);
    CHECK(At(0, 0) == 1.0);
    CHECK(At(1, 1) == 3.0);
    CHECK(At(2, 0) == 2.0);
    CHECK(At(0, 1) == 0.0);
}

TEST_CASE("coomatrix_dynamic_from_transpose_expression",
          "[sparse][coo][dynamic][converting_ctor]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{0, 1}, 2.0}, {{1, 0}, 3.0}, {{1, 1}, 4.0}};
    zipper::COOMatrix<double, 2, 2> A(entries);

    // COOMatrix from a Zipper-wrapped transpose.
    zipper::COOMatrix<double, dyn, dyn> At(A.transpose());

    CHECK(At.rows() == 2);
    CHECK(At.cols() == 2);
    CHECK(At(0, 0) == 1.0);
    CHECK(At(0, 1) == 3.0);
    CHECK(At(1, 0) == 2.0);
    CHECK(At(1, 1) == 4.0);
}

TEST_CASE("csrmatrix_dynamic_from_matrix_product",
          "[sparse][csr][dynamic][converting_ctor]") {
    // A (2x3) * B (3x2) -> C (2x2), materialized into dynamic CSR.
    std::vector<zipper::SparseEntry<double, 2>> a_entries = {
        {{0, 0}, 1.0}, {{0, 2}, 2.0}, {{1, 1}, 3.0}};
    std::vector<zipper::SparseEntry<double, 2>> b_entries = {
        {{0, 0}, 1.0}, {{1, 1}, 1.0}, {{2, 0}, 1.0}};
    zipper::CSRMatrix<double, 2, 3> A(a_entries);
    zipper::CSRMatrix<double, 3, 2> B(b_entries);

    // A * B is a lazy MatrixProduct expression (Zipper-wrapped).
    zipper::CSRMatrix<double, dyn, dyn> C(A * B);

    CHECK(C.rows() == 2);
    CHECK(C.cols() == 2);
    // C = [1*1+0+2*1, 0;  0, 3*1] = [3, 0; 0, 3]
    CHECK(C(0, 0) == 3.0);
    CHECK(C(0, 1) == 0.0);
    CHECK(C(1, 0) == 0.0);
    CHECK(C(1, 1) == 3.0);
}

TEST_CASE("csmatrix_dynamic_from_expression",
          "[sparse][csmatrix][dynamic][converting_ctor]") {
    // CSMatrix has the same converting constructor pattern.
    std::vector<zipper::SparseEntry<double, 2>> entries = {{{0, 0}, 1.0},
                                                           {{1, 1}, 2.0}};
    zipper::CSMatrix<double, 2, 2, zipper::storage::layout_right> A(entries);

    // Construct dynamic-extent CSMatrix from static-extent CSMatrix (Zipper).
    zipper::CSMatrix<double, dyn, dyn, zipper::storage::layout_right> B(A);

    CHECK(B.rows() == 2);
    CHECK(B.cols() == 2);
    CHECK(B(0, 0) == 1.0);
    CHECK(B(1, 1) == 2.0);
    CHECK(B(0, 1) == 0.0);
}

TEST_CASE("csrmatrix_dynamic_from_raw_expression",
          "[sparse][csr][dynamic][converting_ctor]") {
    // Test the Expression converting ctor (not Zipper).
    // Build a static-extent COO accessor and pass it directly.
    zipper::storage::SparseCoordinateAccessor<double, zipper::extents<2, 2>>
        coo_acc;
    coo_acc.emplace(0, 0) = 5.0;
    coo_acc.emplace(1, 1) = 7.0;
    coo_acc.compress();

    // This calls CSRMatrix(const Expression &other) with !is_static.
    zipper::CSRMatrix<double, dyn, dyn> A(coo_acc);

    CHECK(A.rows() == 2);
    CHECK(A.cols() == 2);
    CHECK(A(0, 0) == 5.0);
    CHECK(A(1, 1) == 7.0);
}

TEST_CASE("csrmatrix_dynamic_eval_transpose",
          "[sparse][csr][dynamic][converting_ctor]") {
    // eval() on a transpose of a dynamic-extent CSR should produce a
    // dynamic-extent CSMatrix (since the Swizzle expression has dynamic
    // extents). This exercises the CSMatrix converting ctor from within
    // MatrixBase::eval().
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{0, 2}, 2.0}, {{1, 1}, 3.0}};
    zipper::CSRMatrix<double, 2, 3> A(entries);

    // A.transpose() is a lazy view; eval() materializes it.
    // The result type depends on preferred_layout: CSR transpose -> CSC.
    auto At = A.transpose().eval();

    CHECK(At.rows() == 3);
    CHECK(At.cols() == 2);
    CHECK(At(0, 0) == 1.0);
    CHECK(At(1, 1) == 3.0);
    CHECK(At(2, 0) == 2.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Dynamic-extent sparse operations
// ═══════════════════════════════════════════════════════════════════════════

// ── COO → CSR with dynamic extents ──────────────────────────────────────

TEST_CASE("coo_dynamic_to_csr", "[sparse][coo][csr][dynamic]") {
    constexpr auto dyn = zipper::dynamic_extent;
    zipper::COOMatrix<double, dyn, dyn> A(3, 3);
    A.emplace(0, 0) = 1.0;
    A.emplace(0, 2) = 2.0;
    A.emplace(1, 1) = 3.0;
    A.emplace(2, 0) = 4.0;
    A.emplace(2, 2) = 5.0;
    A.compress();

    auto csr = A.to_csr();
    CHECK(csr.rows() == 3);
    CHECK(csr.cols() == 3);
    CHECK(csr(0, 0) == 1.0);
    CHECK(csr(0, 2) == 2.0);
    CHECK(csr(1, 1) == 3.0);
    CHECK(csr(2, 0) == 4.0);
    CHECK(csr(2, 2) == 5.0);
    CHECK(csr(0, 1) == 0.0);
}

TEST_CASE("coo_dynamic_to_csc", "[sparse][coo][csc][dynamic]") {
    constexpr auto dyn = zipper::dynamic_extent;
    zipper::COOMatrix<double, dyn, dyn> A(3, 3);
    A.emplace(0, 0) = 1.0;
    A.emplace(0, 2) = 2.0;
    A.emplace(1, 1) = 3.0;
    A.emplace(2, 0) = 4.0;
    A.compress();

    auto csc = A.to_csc();
    CHECK(csc.rows() == 3);
    CHECK(csc.cols() == 3);
    CHECK(csc(0, 0) == 1.0);
    CHECK(csc(0, 2) == 2.0);
    CHECK(csc(1, 1) == 3.0);
    CHECK(csc(2, 0) == 4.0);
    CHECK(csc(1, 0) == 0.0);
}

// ── CSR construction from entries with dynamic extents ──────────────────

TEST_CASE("csr_dynamic_from_entries", "[sparse][csr][dynamic]") {
    constexpr auto dyn = zipper::dynamic_extent;
    std::vector<zipper::SparseEntry<double, 2>> entries = {{{0, 0}, 1.0},
                                                           {{0, 2}, 2.0},
                                                           {{1, 1}, 3.0},
                                                           {{2, 0}, 4.0},
                                                           {{2, 2}, 5.0}};
    zipper::CSMatrix<double, dyn, dyn, zipper::storage::layout_right> A(
        entries, zipper::extents<dyn, dyn>(3, 3));

    CHECK(A.rows() == 3);
    CHECK(A.cols() == 3);
    CHECK(A(0, 0) == 1.0);
    CHECK(A(0, 2) == 2.0);
    CHECK(A(1, 1) == 3.0);
    CHECK(A(2, 0) == 4.0);
    CHECK(A(2, 2) == 5.0);
    CHECK(A(1, 0) == 0.0);
}

TEST_CASE("csc_dynamic_from_entries", "[sparse][csc][dynamic]") {
    constexpr auto dyn = zipper::dynamic_extent;
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{0, 2}, 2.0}, {{1, 1}, 3.0}, {{2, 0}, 4.0}};
    zipper::CSMatrix<double, dyn, dyn, zipper::storage::layout_left> A(
        entries, zipper::extents<dyn, dyn>(3, 3));

    CHECK(A.rows() == 3);
    CHECK(A.cols() == 3);
    CHECK(A(0, 0) == 1.0);
    CHECK(A(0, 2) == 2.0);
    CHECK(A(1, 1) == 3.0);
    CHECK(A(2, 0) == 4.0);
}

// ── Roundtrip COO → CSR → COO with dynamic extents ─────────────────────

TEST_CASE("roundtrip_dynamic_coo_csr_coo", "[sparse][roundtrip][dynamic]") {
    constexpr auto dyn = zipper::dynamic_extent;
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{0, 2}, 2.0}, {{1, 1}, 3.0}, {{2, 0}, 4.0}};
    zipper::COOMatrix<double, dyn, dyn> coo(entries,
                                            zipper::extents<dyn, dyn>(3, 3));

    auto csr = coo.to_csr();
    auto coo2 = csr.to_coo();

    CHECK(coo2.rows() == 3);
    CHECK(coo2.cols() == 3);
    for (const auto &e : entries) {
        CHECK(coo2(e.indices[0], e.indices[1]) == e.value);
    }
    CHECK(coo2(0, 1) == 0.0);
    CHECK(coo2(1, 0) == 0.0);
    CHECK(coo2(2, 2) == 0.0);
}

// ── Roundtrip COO → CSC → COO with dynamic extents ─────────────────────

TEST_CASE("roundtrip_dynamic_coo_csc_coo", "[sparse][roundtrip][dynamic]") {
    constexpr auto dyn = zipper::dynamic_extent;
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{0, 2}, 2.0}, {{1, 1}, 3.0}, {{2, 0}, 4.0}};
    zipper::COOMatrix<double, dyn, dyn> coo(entries,
                                            zipper::extents<dyn, dyn>(3, 3));

    auto csc = coo.to_csc();
    auto coo2 = csc.to_coo();

    CHECK(coo2.rows() == 3);
    CHECK(coo2.cols() == 3);
    for (const auto &e : entries) {
        CHECK(coo2(e.indices[0], e.indices[1]) == e.value);
    }
    CHECK(coo2(0, 1) == 0.0);
    CHECK(coo2(1, 0) == 0.0);
    CHECK(coo2(2, 2) == 0.0);
}

// ── SpMV with dynamic extents ───────────────────────────────────────────

TEST_CASE("coo_dynamic_spmv", "[sparse][coo][spmv][dynamic]") {
    // | 1  0  2 |   | 1 |   | 7  |
    // | 0  3  0 | * | 2 | = | 6  |
    // | 4  0  5 |   | 3 |   | 19 |
    constexpr auto dyn = zipper::dynamic_extent;
    zipper::COOMatrix<double, dyn, dyn> A(3, 3);
    A.emplace(0, 0) = 1.0;
    A.emplace(0, 2) = 2.0;
    A.emplace(1, 1) = 3.0;
    A.emplace(2, 0) = 4.0;
    A.emplace(2, 2) = 5.0;
    A.compress();

    zipper::Vector<double, 3> x({1.0, 2.0, 3.0});

    zipper::Vector<double, 3> y(A * x);
    CHECK(y(0) == 7.0);
    CHECK(y(1) == 6.0);
    CHECK(y(2) == 19.0);
}

TEST_CASE("csr_dynamic_spmv", "[sparse][csr][spmv][dynamic]") {
    constexpr auto dyn = zipper::dynamic_extent;
    zipper::COOMatrix<double, dyn, dyn> coo(3, 3);
    coo.emplace(0, 0) = 1.0;
    coo.emplace(0, 2) = 2.0;
    coo.emplace(1, 1) = 3.0;
    coo.emplace(2, 0) = 4.0;
    coo.emplace(2, 2) = 5.0;
    coo.compress();
    auto A = coo.to_csr();

    zipper::Vector<double, 3> x({1.0, 2.0, 3.0});

    zipper::Vector<double, 3> y(A * x);
    CHECK(y(0) == 7.0);
    CHECK(y(1) == 6.0);
    CHECK(y(2) == 19.0);
}

TEST_CASE("csc_dynamic_spmv", "[sparse][csc][spmv][dynamic]") {
    constexpr auto dyn = zipper::dynamic_extent;
    zipper::COOMatrix<double, dyn, dyn> coo(3, 3);
    coo.emplace(0, 0) = 1.0;
    coo.emplace(0, 2) = 2.0;
    coo.emplace(1, 1) = 3.0;
    coo.emplace(2, 0) = 4.0;
    coo.emplace(2, 2) = 5.0;
    coo.compress();
    auto A = coo.to_csc();

    zipper::Vector<double, 3> x({1.0, 2.0, 3.0});

    zipper::Vector<double, 3> y(A * x);
    CHECK(y(0) == 7.0);
    CHECK(y(1) == 6.0);
    CHECK(y(2) == 19.0);
}

// ── SpMV with fully dynamic vector ──────────────────────────────────────

TEST_CASE("csr_dynamic_spmv_dynamic_vector", "[sparse][csr][spmv][dynamic]") {
    constexpr auto dyn = zipper::dynamic_extent;
    zipper::COOMatrix<double, dyn, dyn> coo(3, 3);
    coo.emplace(0, 0) = 1.0;
    coo.emplace(0, 2) = 2.0;
    coo.emplace(1, 1) = 3.0;
    coo.emplace(2, 0) = 4.0;
    coo.emplace(2, 2) = 5.0;
    coo.compress();
    auto A = coo.to_csr();

    zipper::VectorX<double> x(3);
    x(0) = 1.0;
    x(1) = 2.0;
    x(2) = 3.0;

    zipper::VectorX<double> y(A * x);
    CHECK(y(0) == 7.0);
    CHECK(y(1) == 6.0);
    CHECK(y(2) == 19.0);
}

// ── as_csr / as_csc layout conversion with dynamic extents ──────────────

TEST_CASE("csmatrix_dynamic_as_csr_as_csc",
          "[sparse][csmatrix][conversion][dynamic]") {
    constexpr auto dyn = zipper::dynamic_extent;
    zipper::COOMatrix<double, dyn, dyn> coo(3, 3);
    coo.emplace(0, 0) = 1.0;
    coo.emplace(0, 2) = 2.0;
    coo.emplace(1, 1) = 3.0;
    coo.emplace(2, 0) = 4.0;
    coo.emplace(2, 2) = 5.0;
    coo.compress();

    auto csr = coo.to_csr();

    SECTION("as_csc converts dynamic CSR to CSC") {
        auto csc = csr.as_csc();
        CHECK(csc.rows() == 3);
        CHECK(csc.cols() == 3);
        CHECK(csc(0, 0) == 1.0);
        CHECK(csc(0, 2) == 2.0);
        CHECK(csc(1, 1) == 3.0);
        CHECK(csc(2, 0) == 4.0);
        CHECK(csc(2, 2) == 5.0);
    }

    SECTION("as_csr on CSR is identity") {
        auto csr2 = csr.as_csr();
        CHECK(csr2.rows() == 3);
        CHECK(csr2.cols() == 3);
        CHECK(csr2(0, 0) == 1.0);
        CHECK(csr2(2, 2) == 5.0);
    }
}

// ── coeff_ref with dynamic extents ──────────────────────────────────────

TEST_CASE("coo_dynamic_coeff_ref", "[sparse][coo][dynamic]") {
    constexpr auto dyn = zipper::dynamic_extent;
    zipper::COOMatrix<double, dyn, dyn> A(3, 3);
    A.emplace(0, 0) = 1.0;
    A.emplace(1, 1) = 2.0;
    A.compress();

    A.coeff_ref(0, 0) = 10.0;
    CHECK(A(0, 0) == 10.0);

    CHECK_THROWS(A.coeff_ref(0, 1)); // missing entry
}

TEST_CASE("csr_dynamic_coeff_ref", "[sparse][csr][dynamic]") {
    constexpr auto dyn = zipper::dynamic_extent;
    zipper::COOMatrix<double, dyn, dyn> coo(3, 3);
    coo.emplace(0, 0) = 1.0;
    coo.emplace(1, 1) = 2.0;
    coo.compress();
    auto A = coo.to_csr();

    A.coeff_ref(0, 0) = 10.0;
    CHECK(A(0, 0) == 10.0);

    CHECK_THROWS(A.coeff_ref(0, 1)); // missing entry
}

// ── Copy semantics with dynamic extents ─────────────────────────────────

TEST_CASE("coo_dynamic_copy", "[sparse][coo][dynamic]") {
    constexpr auto dyn = zipper::dynamic_extent;
    zipper::COOMatrix<double, dyn, dyn> A(3, 3);
    A.emplace(0, 0) = 1.0;
    A.emplace(1, 2) = 5.0;
    A.compress();

    auto B = A;
    CHECK(B.rows() == 3);
    CHECK(B.cols() == 3);
    CHECK(B(0, 0) == 1.0);
    CHECK(B(1, 2) == 5.0);
}

TEST_CASE("csr_dynamic_copy", "[sparse][csr][dynamic]") {
    constexpr auto dyn = zipper::dynamic_extent;
    zipper::COOMatrix<double, dyn, dyn> coo(3, 3);
    coo.emplace(0, 0) = 1.0;
    coo.emplace(1, 2) = 5.0;
    coo.compress();
    auto A = coo.to_csr();

    auto B = A;
    CHECK(B.rows() == 3);
    CHECK(B.cols() == 3);
    CHECK(B(0, 0) == 1.0);
    CHECK(B(1, 2) == 5.0);
}

// ── Assignment with dynamic extents ─────────────────────────────────────

TEST_CASE("csr_dynamic_assign_from_coo", "[sparse][csr][assignment][dynamic]") {
    constexpr auto dyn = zipper::dynamic_extent;
    zipper::COOMatrix<double, dyn, dyn> coo(3, 3);
    coo.emplace(0, 0) = 1.0;
    coo.emplace(1, 1) = 3.0;
    coo.emplace(2, 0) = 4.0;
    coo.compress();

    zipper::CSMatrix<double, dyn, dyn, zipper::storage::layout_right> csr(3, 3);
    csr = coo;
    CHECK(csr(0, 0) == 1.0);
    CHECK(csr(1, 1) == 3.0);
    CHECK(csr(2, 0) == 4.0);
}

TEST_CASE("coo_dynamic_assign_from_csr", "[sparse][coo][assignment][dynamic]") {
    constexpr auto dyn = zipper::dynamic_extent;
    zipper::COOMatrix<double, dyn, dyn> coo_src(3, 3);
    coo_src.emplace(0, 0) = 1.0;
    coo_src.emplace(1, 1) = 3.0;
    coo_src.compress();
    auto csr = coo_src.to_csr();

    zipper::COOMatrix<double, dyn, dyn> coo_dst(3, 3);
    coo_dst = csr;
    CHECK(coo_dst(0, 0) == 1.0);
    CHECK(coo_dst(1, 1) == 3.0);
}

// ── Non-square dynamic matrices ─────────────────────────────────────────

TEST_CASE("coo_dynamic_nonsquare", "[sparse][coo][dynamic]") {
    constexpr auto dyn = zipper::dynamic_extent;
    zipper::COOMatrix<double, dyn, dyn> A(2, 4);
    A.emplace(0, 0) = 1.0;
    A.emplace(0, 3) = 2.0;
    A.emplace(1, 1) = 3.0;
    A.compress();

    auto csr = A.to_csr();
    CHECK(csr.rows() == 2);
    CHECK(csr.cols() == 4);
    CHECK(csr(0, 0) == 1.0);
    CHECK(csr(0, 3) == 2.0);
    CHECK(csr(1, 1) == 3.0);

    auto coo2 = csr.to_coo();
    CHECK(coo2.rows() == 2);
    CHECK(coo2.cols() == 4);
    CHECK(coo2(0, 0) == 1.0);
    CHECK(coo2(0, 3) == 2.0);
    CHECK(coo2(1, 1) == 3.0);
}

// ── Duplicate entry handling with dynamic extents ───────────────────────

TEST_CASE("coo_dynamic_duplicate_entries", "[sparse][coo][dynamic]") {
    constexpr auto dyn = zipper::dynamic_extent;
    zipper::COOMatrix<double, dyn, dyn> A(3, 3);
    A.emplace(0, 0) = 1.0;
    A.emplace(0, 0) = 2.0;
    A.emplace(1, 1) = 5.0;
    A.compress();

    // Duplicate entries should be summed after compress
    CHECK(A(0, 0) == 3.0);
    CHECK(A(1, 1) == 5.0);
}

// ── Empty dynamic matrix ────────────────────────────────────────────────

TEST_CASE("csr_dynamic_empty_roundtrip", "[sparse][csr][dynamic]") {
    constexpr auto dyn = zipper::dynamic_extent;
    zipper::COOMatrix<double, dyn, dyn> coo(4, 4);
    coo.compress();

    auto csr = coo.to_csr();
    CHECK(csr.rows() == 4);
    CHECK(csr.cols() == 4);

    auto coo2 = csr.to_coo();
    CHECK(coo2.rows() == 4);
    CHECK(coo2.cols() == 4);
    CHECK(coo2(0, 0) == 0.0);
}
