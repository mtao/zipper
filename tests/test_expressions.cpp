#include "catch_include.hpp"
#include <iostream>
#include <zipper/Vector.hpp>
using namespace zipper;
TEST_CASE("test_expression", "[vector][storage][dense][span]") {

    Vector<int,5> v({4,5,3,5,6});

    auto div_3 = v.unary_expr([](int x) -> bool { return x % 3 == 0; });

    REQUIRE(div_3.extent(0) == 5);
    CHECK_FALSE(div_3(0));
    CHECK_FALSE(div_3(1));
    CHECK(div_3(2));
    CHECK_FALSE(div_3(3));
    CHECK(div_3(4));
}
