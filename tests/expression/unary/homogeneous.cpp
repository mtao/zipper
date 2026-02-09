
#include "../../fmt_include.hpp"
#include "../../catch_include.hpp"
#include <iostream>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Random.hpp>
#include <zipper/expression/nullary/Unit.hpp>
#include <zipper/expression/unary/Homogeneous.hpp>

TEST_CASE("test_homogeneous", "[vector][homogeneous]") {
    zipper::Vector a = zipper::expression::nullary::unit_vector<double, 3, 0>();
    zipper::Vector b = zipper::expression::nullary::unit_vector<double>(3, 1);

    auto c = zipper::expression::unary::homogeneous_position(a.expression());
    CHECK(c(0) == a(0));
    CHECK(c(1) == a(1));
    CHECK(c(2) == a(2));
    CHECK(c(3) == 1);
    CHECK(zipper::VectorBase(c) == a.homogeneous());
    auto d = zipper::expression::unary::homogeneous_position(b.expression());
    CHECK(d(0) == b(0));
    CHECK(d(1) == b(1));
    CHECK(d(2) == b(2));
    CHECK(d(3) == 1);
    CHECK(zipper::VectorBase(d) == b.homogeneous());

    auto e = zipper::expression::unary::homogeneous_vector(a.expression());
    CHECK(e(0) == a(0));
    CHECK(e(1) == a(1));
    CHECK(e(2) == a(2));
    CHECK(e(3) == 0);
}
