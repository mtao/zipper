#include "../fmt_include.hpp"
#include "../catch_include.hpp"
#include <iostream>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Random.hpp>

TEST_CASE("test_head_tail", "[vector][homogeneous]") {
    zipper::Vector a = zipper::expression::nullary::uniform_random_view<double>(
        zipper::extents<3>{});
    zipper::Vector b = zipper::expression::nullary::uniform_random_view<double>(
        zipper::create_dextents(3));

    auto c = a.head<2>();
    REQUIRE(c.extent(0) == 2);
    CHECK(c(0) == a(0));
    CHECK(c(1) == a(1));
    auto d = a.tail<2>();
    REQUIRE(d.extent(0) == 2);
    CHECK(d(0) == a(1));
    CHECK(d(1) == a(2));

    auto e = b.head<2>();
    REQUIRE(c.extent(0) == 2);
    CHECK(e(0) == b(0));
    CHECK(e(1) == b(1));
    auto f = b.tail<2>();
    REQUIRE(f.extent(0) == 2);
    CHECK(f(0) == b(1));
    CHECK(f(1) == b(2));
}

// #include <zipper/Vector.hpp>
