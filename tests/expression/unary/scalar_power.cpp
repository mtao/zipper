// Tests for ScalarPower unary expression

#include <cmath>
#include <zipper/Array.hpp>
#include <zipper/ArrayBase.hxx>

#include "../../catch_include.hpp"

TEST_CASE("pow_square", "[pow][unary]") {
    zipper::Array<double, 3> x;
    x(0) = 2.0;
    x(1) = 3.0;
    x(2) = 4.0;

    auto p = x.pow(2.0);
    CHECK(p(0) == 4.0);
    CHECK(p(1) == 9.0);
    CHECK(p(2) == 16.0);
}

TEST_CASE("pow_zero", "[pow][unary]") {
    zipper::Array<double, 3> x;
    x(0) = 2.0;
    x(1) = 3.0;
    x(2) = 4.0;

    auto p = x.pow(0.0);
    CHECK(p(0) == 1.0);
    CHECK(p(1) == 1.0);
    CHECK(p(2) == 1.0);
}

TEST_CASE("pow_one", "[pow][unary]") {
    zipper::Array<double, 3> x;
    x(0) = 2.0;
    x(1) = 3.0;
    x(2) = 4.0;

    auto p = x.pow(1.0);
    CHECK(p(0) == 2.0);
    CHECK(p(1) == 3.0);
    CHECK(p(2) == 4.0);
}

TEST_CASE("pow_fractional", "[pow][unary]") {
    zipper::Array<double, 3> x;
    x(0) = 4.0;
    x(1) = 9.0;
    x(2) = 16.0;

    auto p = x.pow(0.5);
    CHECK(p(0) == Catch::Approx(2.0));
    CHECK(p(1) == Catch::Approx(3.0));
    CHECK(p(2) == Catch::Approx(4.0));
}

TEST_CASE("pow_is_lazy_view", "[pow][unary][view]") {
    zipper::Array<double, 3> x;
    x(0) = 2.0;
    x(1) = 3.0;
    x(2) = 4.0;

    auto p = x.pow(2.0);
    CHECK(p(0) == 4.0);

    x(0) = 5.0;
    CHECK(p(0) == 25.0);
}

TEST_CASE("pow_2d_array", "[pow][unary]") {
    zipper::Array<double, 2, 2> x;
    x(0, 0) = 2.0;
    x(0, 1) = 3.0;
    x(1, 0) = 4.0;
    x(1, 1) = 5.0;

    auto p = x.pow(3.0);
    CHECK(p(0, 0) == 8.0);
    CHECK(p(0, 1) == 27.0);
    CHECK(p(1, 0) == 64.0);
    CHECK(p(1, 1) == 125.0);
}

TEST_CASE("pow_negative_exponent", "[pow][unary]") {
    zipper::Array<double, 2> x;
    x(0) = 2.0;
    x(1) = 4.0;

    auto p = x.pow(-1.0);
    CHECK(p(0) == Catch::Approx(0.5));
    CHECK(p(1) == Catch::Approx(0.25));
}
