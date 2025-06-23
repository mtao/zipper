
#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
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
}
