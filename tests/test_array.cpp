
#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
#include <iostream>
#include <zipper/Array.hpp>




TEST_CASE("test_array_minmax", "[array][storage][dense]") {

    zipper::Array<double,2,2> x;
    x(0,0) = 0;
    x(1,0) = 1;
    x(0,1) = 2;
    x(1,1) = 3;
    zipper::Array<double,2,2> y;
    y(0,0) = 1;
    y(1,0) = 0;
    y(0,1) = -2;
    y(1,1) = 3;

    auto z = min(x,y);

    CHECK(z(0,0) == 0);
    CHECK(z(1,0) == 0);
    CHECK(z(0,1) == -2);
    CHECK(z(1,1) == 3);

    auto v = max(x,y);

    CHECK(v(0,0) == 1);
    CHECK(v(1,0) == 1);
    CHECK(v(0,1) == 2);
    CHECK(v(1,1) == 3);

}
