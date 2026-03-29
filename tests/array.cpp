

#include "catch_include.hpp"
#include <cmath>
#include <iostream>
#include <zipper/Array.hpp>
#include <zipper/ArrayBase.hxx>

TEST_CASE("test_array_minmax", "[array][storage][dense]") {
    zipper::Array<double, 2, 2> x;
    x(0, 0) = 0;
    x(1, 0) = 1;
    x(0, 1) = 2;
    x(1, 1) = 3;
    zipper::Array<double, 2, 2> y;
    y(0, 0) = 1;
    y(1, 0) = 0;
    y(0, 1) = -2;
    y(1, 1) = 3;

    auto z = zipper::min(x, y);

    CHECK(z(0, 0) == 0);
    CHECK(z(1, 0) == 0);
    CHECK(z(0, 1) == -2);
    CHECK(z(1, 1) == 3);

    auto v = zipper::max(x, y);

    CHECK(v(0, 0) == 1);
    CHECK(v(1, 0) == 1);
    CHECK(v(0, 1) == 2);
    CHECK(v(1, 1) == 3);
}

// ============================================================
// Element-wise binary ops between arrays
// ============================================================

TEST_CASE("array_elementwise_add", "[array][binary]") {
    zipper::Array<double, 3> a;
    a(0) = 1.0;
    a(1) = 2.0;
    a(2) = 3.0;
    zipper::Array<double, 3> b;
    b(0) = 10.0;
    b(1) = 20.0;
    b(2) = 30.0;

    auto c = a + b;
    CHECK(c(0) == 11.0);
    CHECK(c(1) == 22.0);
    CHECK(c(2) == 33.0);
}

TEST_CASE("array_elementwise_subtract", "[array][binary]") {
    zipper::Array<double, 3> a;
    a(0) = 10.0;
    a(1) = 20.0;
    a(2) = 30.0;
    zipper::Array<double, 3> b;
    b(0) = 1.0;
    b(1) = 2.0;
    b(2) = 3.0;

    auto c = a - b;
    CHECK(c(0) == 9.0);
    CHECK(c(1) == 18.0);
    CHECK(c(2) == 27.0);
}

TEST_CASE("array_elementwise_multiply", "[array][binary]") {
    zipper::Array<double, 3> a;
    a(0) = 2.0;
    a(1) = 3.0;
    a(2) = 4.0;
    zipper::Array<double, 3> b;
    b(0) = 5.0;
    b(1) = 6.0;
    b(2) = 7.0;

    auto c = a * b;
    CHECK(c(0) == 10.0);
    CHECK(c(1) == 18.0);
    CHECK(c(2) == 28.0);
}

TEST_CASE("array_elementwise_divide", "[array][binary]") {
    zipper::Array<double, 3> a;
    a(0) = 10.0;
    a(1) = 20.0;
    a(2) = 30.0;
    zipper::Array<double, 3> b;
    b(0) = 2.0;
    b(1) = 4.0;
    b(2) = 5.0;

    auto c = a / b;
    CHECK(c(0) == 5.0);
    CHECK(c(1) == 5.0);
    CHECK(c(2) == 6.0);
}

// ============================================================
// Scalar ops on arrays
// ============================================================

TEST_CASE("array_scalar_add", "[array][scalar]") {
    zipper::Array<double, 3> a;
    a(0) = 1.0;
    a(1) = 2.0;
    a(2) = 3.0;

    auto c = a + 10.0;
    CHECK(c(0) == 11.0);
    CHECK(c(1) == 12.0);
    CHECK(c(2) == 13.0);

    auto d = 10.0 + a;
    CHECK(d(0) == 11.0);
    CHECK(d(1) == 12.0);
    CHECK(d(2) == 13.0);
}

TEST_CASE("array_scalar_subtract", "[array][scalar]") {
    zipper::Array<double, 3> a;
    a(0) = 10.0;
    a(1) = 20.0;
    a(2) = 30.0;

    auto c = a - 1.0;
    CHECK(c(0) == 9.0);
    CHECK(c(1) == 19.0);
    CHECK(c(2) == 29.0);

    auto d = 100.0 - a;
    CHECK(d(0) == 90.0);
    CHECK(d(1) == 80.0);
    CHECK(d(2) == 70.0);
}

TEST_CASE("array_scalar_multiply", "[array][scalar]") {
    zipper::Array<double, 3> a;
    a(0) = 1.0;
    a(1) = 2.0;
    a(2) = 3.0;

    auto c = a * 3.0;
    CHECK(c(0) == 3.0);
    CHECK(c(1) == 6.0);
    CHECK(c(2) == 9.0);

    auto d = 3.0 * a;
    CHECK(d(0) == 3.0);
    CHECK(d(1) == 6.0);
    CHECK(d(2) == 9.0);
}

TEST_CASE("array_scalar_divide", "[array][scalar]") {
    zipper::Array<double, 3> a;
    a(0) = 10.0;
    a(1) = 20.0;
    a(2) = 30.0;

    auto c = a / 2.0;
    CHECK(c(0) == 5.0);
    CHECK(c(1) == 10.0);
    CHECK(c(2) == 15.0);
}

// ============================================================
// Comparison ops on arrays
// ============================================================

TEST_CASE("array_comparison_ops", "[array][comparison]") {
    zipper::Array<double, 3> a;
    a(0) = 1.0;
    a(1) = 2.0;
    a(2) = 3.0;

    {
        auto eq = (a == 2.0);
        CHECK_FALSE(eq(0));
        CHECK(eq(1));
        CHECK_FALSE(eq(2));
    }
    {
        auto neq = (a != 2.0);
        CHECK(neq(0));
        CHECK_FALSE(neq(1));
        CHECK(neq(2));
    }
    {
        auto gt = (a > 1.5);
        CHECK_FALSE(gt(0));
        CHECK(gt(1));
        CHECK(gt(2));
    }
    {
        auto lt = (a < 2.5);
        CHECK(lt(0));
        CHECK(lt(1));
        CHECK_FALSE(lt(2));
    }
    {
        auto ge = (a >= 2.0);
        CHECK_FALSE(ge(0));
        CHECK(ge(1));
        CHECK(ge(2));
    }
    {
        auto le = (a <= 2.0);
        CHECK(le(0));
        CHECK(le(1));
        CHECK_FALSE(le(2));
    }
}

// ============================================================
// pow and abs on ArrayBase
// ============================================================

TEST_CASE("array_pow", "[array][unary]") {
    zipper::Array<double, 3> a;
    a(0) = 2.0;
    a(1) = 3.0;
    a(2) = 4.0;

    auto p = a.pow(2.0);
    CHECK(p(0) == 4.0);
    CHECK(p(1) == 9.0);
    CHECK(p(2) == 16.0);
}

TEST_CASE("array_abs", "[array][unary]") {
    zipper::Array<double, 3> a;
    a(0) = -1.0;
    a(1) = 2.0;
    a(2) = -3.0;

    auto ab = a.abs();
    CHECK(ab(0) == 1.0);
    CHECK(ab(1) == 2.0);
    CHECK(ab(2) == 3.0);
}

// ============================================================
// sum and product
// ============================================================

TEST_CASE("array_sum", "[array][reduction]") {
    zipper::Array<double, 4> a;
    a(0) = 1.0;
    a(1) = 2.0;
    a(2) = 3.0;
    a(3) = 4.0;
    CHECK(a.sum() == 10.0);
}

TEST_CASE("array_product", "[array][reduction]") {
    zipper::Array<double, 3> a;
    a(0) = 2.0;
    a(1) = 3.0;
    a(2) = 4.0;
    CHECK(a.product() == 24.0);
}

// ============================================================
// Compound assignment
// ============================================================

TEST_CASE("array_compound_assign", "[array][assign]") {
    zipper::Array<double, 3> a;
    a(0) = 1.0;
    a(1) = 2.0;
    a(2) = 3.0;
    zipper::Array<double, 3> b;
    b(0) = 10.0;
    b(1) = 20.0;
    b(2) = 30.0;

    a += b;
    CHECK(a(0) == 11.0);
    CHECK(a(1) == 22.0);
    CHECK(a(2) == 33.0);

    a -= b;
    CHECK(a(0) == 1.0);
    CHECK(a(1) == 2.0);
    CHECK(a(2) == 3.0);

    a *= 2.0;
    CHECK(a(0) == 2.0);
    CHECK(a(1) == 4.0);
    CHECK(a(2) == 6.0);

    a /= 2.0;
    CHECK(a(0) == 1.0);
    CHECK(a(1) == 2.0);
    CHECK(a(2) == 3.0);
}

// ============================================================
// Negate
// ============================================================

TEST_CASE("array_negate", "[array][unary]") {
    zipper::Array<double, 3> a;
    a(0) = 1.0;
    a(1) = -2.0;
    a(2) = 3.0;

    auto neg = -a;
    CHECK(neg(0) == -1.0);
    CHECK(neg(1) == 2.0);
    CHECK(neg(2) == -3.0);
}

// ============================================================
// normalized
// ============================================================

TEST_CASE("array_normalized", "[array][norm]") {
    zipper::Array<double, 3> a;
    a(0) = 3.0;
    a(1) = 4.0;
    a(2) = 0.0;

    auto n = a.normalized();
    CHECK(n(0) == Catch::Approx(0.6));
    CHECK(n(1) == Catch::Approx(0.8));
    CHECK(n(2) == Catch::Approx(0.0));
}

// ============================================================
// Element-wise three-way comparison (operator<=>)
// ============================================================

TEST_CASE("array_elementwise_spaceship", "[array][comparison][spaceship]") {
    zipper::Array<double, 4> a;
    a(0) = 1.0;
    a(1) = 5.0;
    a(2) = 3.0;
    a(3) = 7.0;
    zipper::Array<double, 4> b;
    b(0) = 2.0;
    b(1) = 3.0;
    b(2) = 3.0;
    b(3) = 1.0;

    auto cmp = (a <=> b);
    CHECK(cmp(0) < 0); // 1 < 2
    CHECK(cmp(1) > 0); // 5 > 3
    CHECK(cmp(2) == 0); // 3 == 3
    CHECK(cmp(3) > 0); // 7 > 1
}

TEST_CASE("array_scalar_spaceship", "[array][comparison][spaceship]") {
    zipper::Array<double, 4> a;
    a(0) = 1.0;
    a(1) = 5.0;
    a(2) = 3.0;
    a(3) = 7.0;

    auto cmp = (a <=> 3.0);
    CHECK(cmp(0) < 0); // 1 < 3
    CHECK(cmp(1) > 0); // 5 > 3
    CHECK(cmp(2) == 0); // 3 == 3
    CHECK(cmp(3) > 0); // 7 > 3
}

// ============================================================
// Bool select: mask.select(true_val, false_val)
// ============================================================

TEST_CASE("array_bool_select", "[array][select]") {
    zipper::Array<double, 4> a;
    a(0) = 1.0;
    a(1) = 5.0;
    a(2) = 3.0;
    a(3) = 7.0;

    // Create a bool mask: a > 3.0
    auto mask = (a > 3.0);
    CHECK_FALSE(mask(0));
    CHECK(mask(1));
    CHECK_FALSE(mask(2));
    CHECK(mask(3));

    zipper::Array<double, 4> tv;
    tv(0) = 100.0;
    tv(1) = 200.0;
    tv(2) = 300.0;
    tv(3) = 400.0;
    zipper::Array<double, 4> fv;
    fv(0) = -1.0;
    fv(1) = -2.0;
    fv(2) = -3.0;
    fv(3) = -4.0;

    auto result = mask.select(tv, fv);
    CHECK(result(0) == -1.0); // false -> fv
    CHECK(result(1) == 200.0); // true  -> tv
    CHECK(result(2) == -3.0); // false -> fv
    CHECK(result(3) == 400.0); // true  -> tv
}

TEST_CASE("array_bool_select_all_true", "[array][select]") {
    zipper::Array<double, 3> a;
    a(0) = 10.0;
    a(1) = 20.0;
    a(2) = 30.0;

    auto mask = (a > 0.0); // all true

    zipper::Array<double, 3> tv;
    tv(0) = 1.0;
    tv(1) = 2.0;
    tv(2) = 3.0;
    zipper::Array<double, 3> fv;
    fv(0) = -1.0;
    fv(1) = -2.0;
    fv(2) = -3.0;

    auto result = mask.select(tv, fv);
    CHECK(result(0) == 1.0);
    CHECK(result(1) == 2.0);
    CHECK(result(2) == 3.0);
}

TEST_CASE("array_bool_select_all_false", "[array][select]") {
    zipper::Array<double, 3> a;
    a(0) = -10.0;
    a(1) = -20.0;
    a(2) = -30.0;

    auto mask = (a > 0.0); // all false

    zipper::Array<double, 3> tv;
    tv(0) = 1.0;
    tv(1) = 2.0;
    tv(2) = 3.0;
    zipper::Array<double, 3> fv;
    fv(0) = -1.0;
    fv(1) = -2.0;
    fv(2) = -3.0;

    auto result = mask.select(tv, fv);
    CHECK(result(0) == -1.0);
    CHECK(result(1) == -2.0);
    CHECK(result(2) == -3.0);
}

// ============================================================
// Ordering select: ordering.select(less, equal, greater)
// ============================================================

TEST_CASE("array_ordering_select", "[array][select][spaceship]") {
    zipper::Array<double, 4> a;
    a(0) = 1.0;
    a(1) = 5.0;
    a(2) = 3.0;
    a(3) = 7.0;
    zipper::Array<double, 4> b;
    b(0) = 2.0;
    b(1) = 3.0;
    b(2) = 3.0;
    b(3) = 1.0;

    auto ord = (a <=> b); // <0, >0, ==0, >0

    zipper::Array<double, 4> less_vals;
    less_vals(0) = -10.0;
    less_vals(1) = -20.0;
    less_vals(2) = -30.0;
    less_vals(3) = -40.0;
    zipper::Array<double, 4> equal_vals;
    equal_vals(0) = 0.0;
    equal_vals(1) = 0.0;
    equal_vals(2) = 0.0;
    equal_vals(3) = 0.0;
    zipper::Array<double, 4> greater_vals;
    greater_vals(0) = 10.0;
    greater_vals(1) = 20.0;
    greater_vals(2) = 30.0;
    greater_vals(3) = 40.0;

    auto result = ord.select(less_vals, equal_vals, greater_vals);
    CHECK(result(0) == -10.0); // 1 < 2 -> less
    CHECK(result(1) == 20.0); // 5 > 3 -> greater
    CHECK(result(2) == 0.0); // 3 == 3 -> equal
    CHECK(result(3) == 40.0); // 7 > 1 -> greater
}

TEST_CASE("array_ordering_select_scalar_spaceship",
          "[array][select][spaceship]") {
    zipper::Array<double, 5> a;
    a(0) = 1.0;
    a(1) = 3.0;
    a(2) = 5.0;
    a(3) = 7.0;
    a(4) = 3.0;

    // Compare against scalar 3.0
    auto ord = (a <=> 3.0);

    zipper::Array<double, 5> less_vals;
    less_vals(0) = -1.0;
    less_vals(1) = -1.0;
    less_vals(2) = -1.0;
    less_vals(3) = -1.0;
    less_vals(4) = -1.0;
    zipper::Array<double, 5> equal_vals;
    equal_vals(0) = 0.0;
    equal_vals(1) = 0.0;
    equal_vals(2) = 0.0;
    equal_vals(3) = 0.0;
    equal_vals(4) = 0.0;
    zipper::Array<double, 5> greater_vals;
    greater_vals(0) = 1.0;
    greater_vals(1) = 1.0;
    greater_vals(2) = 1.0;
    greater_vals(3) = 1.0;
    greater_vals(4) = 1.0;

    auto result = ord.select(less_vals, equal_vals, greater_vals);
    CHECK(result(0) == -1.0); // 1 < 3 -> less
    CHECK(result(1) == 0.0); // 3 == 3 -> equal
    CHECK(result(2) == 1.0); // 5 > 3 -> greater
    CHECK(result(3) == 1.0); // 7 > 3 -> greater
    CHECK(result(4) == 0.0); // 3 == 3 -> equal
}

// ============================================================
// Select chaining: use select result in further computation
// ============================================================

TEST_CASE("array_select_chaining", "[array][select]") {
    zipper::Array<double, 3> a;
    a(0) = -5.0;
    a(1) = 0.0;
    a(2) = 5.0;

    // Clamp negative values to 0 using bool select
    auto mask = (a > 0.0);
    zipper::Array<double, 3> zeros;
    zeros(0) = 0.0;
    zeros(1) = 0.0;
    zeros(2) = 0.0;

    auto clamped = mask.select(a, zeros);
    CHECK(clamped(0) == 0.0);
    CHECK(clamped(1) == 0.0);
    CHECK(clamped(2) == 5.0);
}
