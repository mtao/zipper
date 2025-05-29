#include <catch2/catch_all.hpp>
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
    storage::SparseCoordinateAccessor<double, extents<5, 5>> A;
    A.emplace(3, 3) = 2;
    A.emplace(4, 3) = 3;
    A.emplace(1, 3) = 4;
    A.emplace(4, 4) = 7;
    A.emplace(0, 3) = 9;

    // try without comrpessing to make sure hte internal compression does
    // something
    auto AC = storage::detail::to_sparse_compressed_data(A);
    CHECK(AC.coeff(0, 0) == 0);
    CHECK(AC.coeff(3, 3) == 2);
    CHECK(AC.coeff(4, 3) == 3);
    CHECK(AC.coeff(1, 3) == 4);
    CHECK(AC.coeff(4, 4) == 7);
    CHECK(AC.coeff(0, 3) == 9);

    A.compress();
    CHECK(A.coeff(3, 3) == 2);
    CHECK(A.coeff(4, 3) == 3);
    CHECK(A.coeff(1, 3) == 4);
    CHECK(A.coeff(4, 4) == 7);
    CHECK(A.coeff(0, 3) == 9);

    auto BC = storage::detail::to_sparse_compressed_data(A);
    CHECK(BC.coeff(0, 0) == 0);
    CHECK(BC.coeff(3, 3) == 2);
    CHECK(BC.coeff(4, 3) == 3);
    CHECK(BC.coeff(1, 3) == 4);
    CHECK(BC.coeff(4, 4) == 7);
    CHECK(BC.coeff(0, 3) == 9);
}
