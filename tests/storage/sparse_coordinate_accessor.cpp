

#include <catch2/catch_all.hpp>
#include <zipper/storage/SparseCoordinateAccessor.hpp>

// #include "zipper/storage/SpanStorage.hpp"

using namespace zipper;
TEST_CASE("sparse_coordinate_accessor", "[sparse]") {
    auto checker = [](auto& A) {
        A.emplace(3, 3) = 2;
        A.emplace(4, 3) = 3;
        REQUIRE(!A.is_compressed());
        A.emplace(1, 3) = 4;
        A.emplace(4, 4) = 7;
        A.emplace(0, 3) = 9;
        REQUIRE(!A.is_compressed());
        A.compress();
        REQUIRE(A.is_compressed());
        CHECK(A.coeff(3, 2) == 0);
        CHECK(A.coeff_ref(3, 3) == 2);
        CHECK(A.coeff_ref(4, 3) == 3);
        CHECK(A.coeff(3, 3) == 2);
        CHECK(A.coeff(4, 3) == 3);
        CHECK(A.coeff(1, 3) == 4);
        CHECK(A.coeff(4, 4) == 7);
        CHECK(A.coeff(0, 3) == 9);
        A.emplace(0, 3) = 9;
        REQUIRE(!A.is_compressed());
        A.emplace(1, 3) = 5;
        A.emplace(0, 3) = 9;
        A.compress();
        CHECK(A.coeff(1, 3) == 9);
        CHECK(A.coeff(0, 3) == 27);
    };
    storage::SparseCoordinateAccessor<double, extents<5, 5>> A;
    checker(A);
    storage::SparseCoordinateAccessor<double, dextents<2>> B(
        create_dextents(5, 5));
    checker(B);
}
