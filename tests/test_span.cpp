
#include "catch_include.hpp"
#include <iostream>
#include <span>
#include <zipper/VectorBase.hpp>
using namespace zipper;
TEST_CASE("test_vector_span", "[vector][storage][dense][span]") {
    std::vector<double> vec = {2, 3};
    VectorBase v = std::span<double, 2>(vec);

    static_assert(v.static_extent(0) == 2);
    REQUIRE(v.extent(0) == 2);
    CHECK(v(0) == 2);
    CHECK(v(1) == 3);

    std::array<double, 2> y;
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
