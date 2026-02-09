// Tests for Abs unary expression

#include <zipper/Array.hpp>
#include <zipper/Vector.hpp>
#include <zipper/ArrayBase.hxx>

#include "../../catch_include.hpp"

TEST_CASE("abs_positive_values", "[abs][unary]") {
    zipper::Array<double, 3> x;
    x(0) = 1.0;
    x(1) = 2.0;
    x(2) = 3.0;

    auto a = x.abs();
    CHECK(a(0) == 1.0);
    CHECK(a(1) == 2.0);
    CHECK(a(2) == 3.0);
}

TEST_CASE("abs_negative_values", "[abs][unary]") {
    zipper::Array<double, 3> x;
    x(0) = -1.0;
    x(1) = -2.5;
    x(2) = -0.5;

    auto a = x.abs();
    CHECK(a(0) == 1.0);
    CHECK(a(1) == 2.5);
    CHECK(a(2) == 0.5);
}

TEST_CASE("abs_mixed_values", "[abs][unary]") {
    zipper::Array<double, 4> x;
    x(0) = -3.0;
    x(1) = 0.0;
    x(2) = 4.0;
    x(3) = -5.0;

    auto a = x.abs();
    CHECK(a(0) == 3.0);
    CHECK(a(1) == 0.0);
    CHECK(a(2) == 4.0);
    CHECK(a(3) == 5.0);
}

TEST_CASE("abs_is_lazy_view", "[abs][unary][view]") {
    zipper::Array<double, 3> x;
    x(0) = -1.0;
    x(1) = -2.0;
    x(2) = -3.0;

    auto a = x.abs();
    CHECK(a(0) == 1.0);

    // abs() is a view — changing x changes the result
    x(0) = -10.0;
    CHECK(a(0) == 10.0);
}

TEST_CASE("abs_2d_array", "[abs][unary]") {
    zipper::Array<double, 2, 2> x;
    x(0, 0) = -1.0;
    x(0, 1) = 2.0;
    x(1, 0) = -3.0;
    x(1, 1) = 4.0;

    auto a = x.abs();
    CHECK(a(0, 0) == 1.0);
    CHECK(a(0, 1) == 2.0);
    CHECK(a(1, 0) == 3.0);
    CHECK(a(1, 1) == 4.0);
}
