
#include "catch_include.hpp"
#include <iostream>
#include <span>
#include <zipper/VectorBase.hpp>
#include <zipper/Vector.hpp>
#include <zipper/Matrix.hpp>
using namespace zipper;
TEST_CASE("test_vector_span", "[vector][storage][dense][span]") {
    std::vector<int> vec = {2, 3};
    VectorBase v = std::span<int, 2>(vec);
    VectorBase v_const = std::span<const int, 2>(vec);

    auto c = v_const.eval();
    CHECK((c == v_const));

    static_assert(v.static_extent(0) == 2);
    REQUIRE(v.extent(0) == 2);
    CHECK(v(0) == 2);
    CHECK(v(1) == 3);

    std::array<int, 2> y;
    VectorBase z(y);
    z = v.view();
    CHECK(y[0] == 2);
    CHECK(y[1] == 3);

    z(0) = 3;
    z(1) = 4;

    CHECK(y[0] == 3);
    CHECK(y[1] == 4);
    z = v;
    CHECK(y[0] == 2);
    CHECK(y[1] == 3);

    z(0) = 3;
    z(1) = 4;

    CHECK(y[0] == 3);
    CHECK(y[1] == 4);

    // this last case WOULD be very cool, but seems to not work due to a parse
    // limitation in type deductions? In particular, gcc at least seems to
    // really want y to be the name of a variable of type VectorBase
    // VectorBase(y) = {4, 5};
    // CHECK(v(0) == 2);
    // CHECK(v(1) == 3);
}
TEST_CASE("test_matrix_span", "[matrix][storage][dense][span]") {
    std::vector<int> vec = {0,1,2,3};
    zipper::Matrix<int,2,2>::span_type M = std::span<int,4>(vec);
    zipper::Matrix<int,std::dynamic_extent,std::dynamic_extent>::span_type Md(std::span<int>(vec), zipper::create_dextents(2,2));

    static_assert(M.static_extent(0) == 2);
    static_assert(M.static_extent(1) == 2);
    static_assert(Md.static_extent(0) == std::dynamic_extent);
    static_assert(Md.static_extent(1) == std::dynamic_extent);
    REQUIRE(M.extent(0) == 2);
    REQUIRE(M.extent(1) == 2);
    REQUIRE(Md.extent(0) == 2);
    REQUIRE(Md.extent(1) == 2);

    CHECK(M(0,0) == 0);
    CHECK(M(0,1) == 1);
    CHECK(M(1,0) == 2);
    CHECK(M(1,1) == 3);

    CHECK((M == Md));


    // this last case WOULD be very cool, but seems to not work due to a parse
    // limitation in type deductions? In particular, gcc at least seems to
    // really want y to be the name of a variable of type VectorBase
    // VectorBase(y) = {4, 5};
    // CHECK(v(0) == 2);
    // CHECK(v(1) == 3);
}
