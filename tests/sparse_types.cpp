
#include <vector>

#include <zipper/COOMatrix.hpp>
#include <zipper/COOVector.hpp>
#include <zipper/CSRMatrix.hpp>
#include <zipper/CSRVector.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/VectorBase.hxx>
#include <zipper/MatrixBase.hxx>

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
    zipper::COOMatrix<double, zipper::dynamic_extent, zipper::dynamic_extent> A(4, 5);
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
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 1}, 3.0}, {{2, 0}, 4.0}};
    constexpr auto dyn = zipper::dynamic_extent;
    zipper::COOMatrix<double, dyn, dyn> A(entries, zipper::extents<dyn, dyn>(3, 3));

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
    std::vector<zipper::SparseEntry<double, 1>> entries = {
        {{2}, 3.14}, {{4}, 2.72}};
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
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 1}, 7.0}, {{1, 0}, 8.0}};
    zipper::COOMatrix<double, 3, 3> coo(entries);

    auto csr = coo.to_csr();
    CHECK(csr(0, 1) == 7.0);
    CHECK(csr(1, 0) == 8.0);
    CHECK(csr(0, 0) == 0.0);
}

TEST_CASE("csr_matrix_coeff_ref", "[sparse][csr][matrix]") {
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{1, 1}, 2.0}};
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
    std::vector<zipper::SparseEntry<double, 1>> entries = {
        {{0}, 1.0}, {{4}, 5.0}};
    zipper::CSRVector<double, 10> csr(entries);

    CHECK(csr(0) == 1.0);
    CHECK(csr(4) == 5.0);
    CHECK(csr(2) == 0.0);
}

TEST_CASE("csr_vector_to_csr_via_method", "[sparse][csr][vector]") {
    std::vector<zipper::SparseEntry<double, 1>> entries = {
        {{2}, 3.0}, {{7}, 9.0}};
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

    for (const auto &e : entries) {
        CHECK(coo2(e.indices[0]) == e.value);
    }
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
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{0, 2}, 2.0},
        {{1, 1}, 3.0},
        {{2, 0}, 4.0}, {{2, 2}, 5.0}};
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
    std::vector<zipper::SparseEntry<double, 2>> entries = {
        {{0, 0}, 1.0}, {{1, 1}, 2.0}};
    zipper::CSRMatrix<double, 3, 3> A(entries);

    auto B = A;
    CHECK(B(0, 0) == 1.0);
    CHECK(B(1, 1) == 2.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  CTAD
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("coo_matrix_ctad", "[sparse][coo][matrix]") {
    zipper::storage::SparseCoordinateAccessor<double, zipper::extents<3, 3>> acc;
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
