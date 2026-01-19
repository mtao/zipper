
#include "../fmt_include.hpp"
#include "../catch_include.hpp"
#include <iostream>
#include <zipper/Vector.hpp>
#include <zipper/views/nullary/ConstantView.hpp>
#include <zipper/views/nullary/RandomView.hpp>
#include <zipper/views/nullary/UnitView.hpp>
#include <zipper/views/unary/HomogeneousView.hpp>
#include <zipper/views/unary/PartialTraceView.hpp>

TEST_CASE("test_homogeneous", "[vector][homogeneous]") {
    zipper::Vector a = zipper::views::nullary::unit_vector<double, 3, 0>();
    zipper::Vector b = zipper::views::nullary::unit_vector<double>(3, 1);

    auto c = zipper::views::unary::homogeneous_position(a.view());
    CHECK(c(0) == a(0));
    CHECK(c(1) == a(1));
    CHECK(c(2) == a(2));
    CHECK(c(3) == 1);
    CHECK(zipper::VectorBase(c) == a.homogeneous());
    auto d = zipper::views::unary::homogeneous_position(b.view());
    CHECK(d(0) == b(0));
    CHECK(d(1) == b(1));
    CHECK(d(2) == b(2));
    CHECK(d(3) == 1);
    CHECK(zipper::VectorBase(d) == b.homogeneous());

    auto e = zipper::views::unary::homogeneous_vector(a.view());
    CHECK(e(0) == a(0));
    CHECK(e(1) == a(1));
    CHECK(e(2) == a(2));
    CHECK(e(3) == 0);
}

// #include <zipper/Vector.hpp>
