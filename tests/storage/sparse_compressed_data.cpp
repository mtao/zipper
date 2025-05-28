#include <catch2/catch_all.hpp>
#include <zipper/storage/detail/SparseCompressedData.hpp>

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
