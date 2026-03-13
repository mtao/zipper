#include "../catch_include.hpp"
#include <zipper/storage/SparseCoordinateAccessor.hpp>
#include <zipper/storage/detail/SparseCompressedData.hpp>
#include <zipper/storage/detail/to_sparse_compressed_data.hpp>

// #include "zipper/storage/SpanStorage.hpp"

using namespace zipper;
TEST_CASE("sparse_compressed_vector_data", "[sparse]") {
    zipper::storage::detail::SparseCompressedData<double, 0> vec;

    vec.insert_back(0) = 4;
    vec.insert_back(3) = 6;
    CHECK(vec.coeff(0) == 4);
    CHECK(vec.coeff(1) == 0);
    CHECK(vec.coeff(3) == 6);
}
TEST_CASE("sparse_compressed_matrix_data", "[sparse]") {
    zipper::storage::detail::SparseCompressedData<double, 1> mat;

    mat.insert_back(0, 1) = 4;
    mat.insert_back(3, 2) = 6;
    CHECK(mat.coeff(0, 0) == 0);
    CHECK(mat.coeff(0, 1) == 4);
    CHECK(mat.coeff(2, 2) == 0);
    CHECK(mat.coeff(3, 2) == 6);
}
TEST_CASE("sparse_compressed_3tensor_data", "[sparse]") {
    zipper::storage::detail::SparseCompressedData<double, 2> mat;

    mat.insert_back(0, 1, 2) = 4;
    mat.insert_back(3, 2, 2) = 6;
    CHECK(mat.coeff(0, 0, 2) == 0);
    CHECK(mat.coeff(0, 1, 2) == 4);
    CHECK(mat.coeff(2, 2, 2) == 0);
    CHECK(mat.coeff(3, 2, 2) == 6);
}

TEST_CASE("sparse_compressed_data_from_coordinate_data", "[sparse]") {
    // filling the sparse coordinate data
    storage::SparseCoordinateAccessor<double, extents<5, 5>> A;
    A.emplace(3, 3) = 2;
    A.emplace(4, 3) = 3;
    A.emplace(1, 3) = 4;
    A.emplace(4, 4) = 7;
    A.emplace(0, 3) = 9;

    // try without compressing to make sure the internal compression does
    // something
    auto AC = storage::detail::to_sparse_compressed_data(A);
    CHECK(AC.coeff(0, 0) == 0);
    CHECK(AC.coeff(3, 3) == 2);
    CHECK(AC.coeff(4, 3) == 3);
    CHECK(AC.coeff(1, 3) == 4);
    CHECK(AC.coeff(4, 4) == 7);
    CHECK(AC.coeff(0, 3) == 9);

    // double checking the sparse coordinate is right
    A.compress();
    CHECK(A.coeff(3, 3) == 2);
    CHECK(A.coeff(4, 3) == 3);
    CHECK(A.coeff(1, 3) == 4);
    CHECK(A.coeff(4, 4) == 7);
    CHECK(A.coeff(0, 3) == 9);

    // checking that it works with compressed data
    auto BC = storage::detail::to_sparse_compressed_data(A);
    CHECK(BC.coeff(0, 0) == 0);
    CHECK(BC.coeff(3, 3) == 2);
    CHECK(BC.coeff(4, 3) == 3);
    CHECK(BC.coeff(1, 3) == 4);
    CHECK(BC.coeff(4, 4) == 7);
    CHECK(BC.coeff(0, 3) == 9);

    // try reversing indices
    auto CC = storage::detail::to_sparse_compressed_data<true>(A);
    CHECK(CC.coeff(0, 0) == 0);
    CHECK(CC.coeff(3, 3) == 2);
    CHECK(CC.coeff(3, 4) == 3);
    CHECK(CC.coeff(3, 1) == 4);
    CHECK(CC.coeff(4, 4) == 7);
    CHECK(CC.coeff(3, 0) == 9);
}

// --- Edge case tests ---

TEST_CASE("sparse_compressed_empty_coeff", "[sparse]") {
    // coeff() on empty SparseCompressedData should return 0
    zipper::storage::detail::SparseCompressedData<double, 0> vec;
    CHECK(vec.coeff(0) == 0.0);
    CHECK(vec.coeff(42) == 0.0);

    zipper::storage::detail::SparseCompressedData<double, 1> mat;
    CHECK(mat.coeff(0, 0) == 0.0);
    CHECK(mat.coeff(5, 3) == 0.0);
}

TEST_CASE("sparse_compressed_single_element", "[sparse]") {
    zipper::storage::detail::SparseCompressedData<double, 0> vec;
    vec.insert_back(5) = 42.0;
    CHECK(vec.coeff(5) == 42.0);
    CHECK(vec.coeff(0) == 0.0);
    CHECK(vec.coeff(4) == 0.0);
    CHECK(vec.coeff(6) == 0.0);
    CHECK(vec.size() == 1);
}

TEST_CASE("sparse_compressed_out_of_order_throws", "[sparse]") {
    // At rank N>=1, out-of-order outer index should throw
    zipper::storage::detail::SparseCompressedData<double, 1> mat;
    mat.insert_back(3, 0) = 1.0;
    CHECK_THROWS_AS(mat.insert_back(1, 0), std::invalid_argument);
}

TEST_CASE("sparse_compressed_duplicate_outer_index", "[sparse]") {
    // Same outer index should be allowed (extends the span)
    zipper::storage::detail::SparseCompressedData<double, 1> mat;
    mat.insert_back(2, 0) = 10.0;
    mat.insert_back(2, 3) = 20.0;
    mat.insert_back(5, 1) = 30.0;

    CHECK(mat.coeff(2, 0) == 10.0);
    CHECK(mat.coeff(2, 3) == 20.0);
    CHECK(mat.coeff(5, 1) == 30.0);
    CHECK(mat.coeff(2, 1) == 0.0); // not present
    CHECK(mat.coeff(0, 0) == 0.0); // not present
}

TEST_CASE("sparse_compressed_many_elements", "[sparse]") {
    // Larger data set
    zipper::storage::detail::SparseCompressedData<double, 0> vec;
    for (zipper::index_type i = 0; i < 100; ++i) {
        vec.insert_back(i * 2) = static_cast<double>(i);
    }
    CHECK(vec.size() == 100);
    CHECK(vec.coeff(0) == 0.0);
    CHECK(vec.coeff(2) == 1.0);
    CHECK(vec.coeff(198) == 99.0);
    CHECK(vec.coeff(1) == 0.0);  // odd index, not present
    CHECK(vec.coeff(199) == 0.0); // past last inserted
}

TEST_CASE("sparse_compressed_integer_type", "[sparse]") {
    zipper::storage::detail::SparseCompressedData<int, 0> vec;
    vec.insert_back(0) = 42;
    vec.insert_back(5) = -7;
    CHECK(vec.coeff(0) == 42);
    CHECK(vec.coeff(5) == -7);
    CHECK(vec.coeff(3) == 0);
}

TEST_CASE("sparse_compressed_copy_move", "[sparse]") {
    zipper::storage::detail::SparseCompressedData<double, 0> vec;
    vec.insert_back(0) = 1.0;
    vec.insert_back(3) = 2.0;

    // Copy
    auto vec2 = vec;
    CHECK(vec2.coeff(0) == 1.0);
    CHECK(vec2.coeff(3) == 2.0);

    // Move
    auto vec3 = std::move(vec2);
    CHECK(vec3.coeff(0) == 1.0);
    CHECK(vec3.coeff(3) == 2.0);
}

TEST_CASE("sparse_compressed_conversion_empty", "[sparse]") {
    // Convert empty coordinate accessor to compressed
    storage::SparseCoordinateAccessor<double, extents<5, 5>> A;
    A.compress();
    auto C = storage::detail::to_sparse_compressed_data(A);
    CHECK(C.coeff(0, 0) == 0.0);
    CHECK(C.coeff(4, 4) == 0.0);
}

TEST_CASE("sparse_compressed_conversion_rank1", "[sparse]") {
    storage::SparseCoordinateAccessor<double, extents<10>> A;
    A.emplace(3) = 5.0;
    A.emplace(7) = 9.0;
    A.compress();
    auto C = storage::detail::to_sparse_compressed_data(A);
    CHECK(C.coeff(3) == 5.0);
    CHECK(C.coeff(7) == 9.0);
    CHECK(C.coeff(0) == 0.0);
    CHECK(C.coeff(9) == 0.0);
}

TEST_CASE("sparse_compressed_conversion_rank3", "[sparse]") {
    // Note: SparseCompressedData::insert_back_ requires globally non-decreasing
    // indices at every level, so we choose entries whose inner indices are
    // non-decreasing in insertion (lexicographic) order.
    storage::SparseCoordinateAccessor<double, extents<2, 3, 4>> A;
    A.emplace(0, 1, 2) = 3.0;
    A.emplace(1, 2, 3) = 7.0;
    A.compress();
    auto C = storage::detail::to_sparse_compressed_data(A);
    CHECK(C.coeff(0, 1, 2) == 3.0);
    CHECK(C.coeff(1, 2, 3) == 7.0);
    CHECK(C.coeff(0, 0, 0) == 0.0);
    CHECK(C.coeff(1, 1, 1) == 0.0);
}

TEST_CASE("sparse_compressed_conversion_with_duplicates", "[sparse]") {
    // Duplicates in coordinate accessor should be accumulated through conversion
    storage::SparseCoordinateAccessor<double, extents<5, 5>> A;
    A.emplace(2, 3) = 1.0;
    A.emplace(2, 3) = 4.0;
    // Don't compress -- let to_sparse_compressed_data handle it
    REQUIRE(!A.is_compressed());
    auto C = storage::detail::to_sparse_compressed_data(A);
    CHECK(C.coeff(2, 3) == 5.0); // 1.0 + 4.0
    CHECK(C.coeff(0, 0) == 0.0);
}

TEST_CASE("sparse_compressed_conversion_dynamic_extents", "[sparse]") {
    storage::SparseCoordinateAccessor<double, dextents<2>> A(
        create_dextents(4, 6));
    A.emplace(0, 5) = 1.0;
    A.emplace(3, 0) = 2.0;
    A.compress();
    auto C = storage::detail::to_sparse_compressed_data(A);
    CHECK(C.coeff(0, 5) == 1.0);
    CHECK(C.coeff(3, 0) == 2.0);
    CHECK(C.coeff(0, 0) == 0.0);
}
