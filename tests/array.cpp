

#include "catch_include.hpp"
#include <cmath>
#include <iostream>
#include <zipper/Array.hpp>
#include <zipper/ArrayBase.hxx>
#include <zipper/Vector.hpp>

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

TEST_CASE("array_normalized_owns_rvalue_expression", "[array][norm]") {
    zipper::Array<double, 3> a;
    a(0) = 3.0;
    a(1) = 4.0;
    a(2) = 0.0;
    zipper::Array<double, 3> b;
    b(0) = 0.0;
    b(1) = 0.0;
    b(2) = 12.0;

    // The normalized expression must own the temporary a + b expression.
    auto n = (a + b).normalized();
    CHECK(n(0) == Catch::Approx(3.0 / 13.0));
    CHECK(n(1) == Catch::Approx(4.0 / 13.0));
    CHECK(n(2) == Catch::Approx(12.0 / 13.0));
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

TEST_CASE("array_select_scalars", "[array][select]") {
    zipper::Array<double, 4> a;
    a(0) = 1.0;
    a(1) = 5.0;
    a(2) = 3.0;
    a(3) = 7.0;
    zipper::Array<double, 4> b;
    b(0) = -1.0;
    b(1) = -2.0;
    b(2) = -3.0;
    b(3) = -4.0;

    SECTION("mixed scalar and array branches") {
        auto clamped = (a <=> 4.0).select(a, 4.0, 4.0); // min(a, 4)
        CHECK(clamped(0) == 1.0);
        CHECK(clamped(1) == 4.0);
        CHECK(clamped(2) == 3.0);
        CHECK(clamped(3) == 4.0);

        auto mask = (a == 3.0);
        auto picked = mask.select(1.0, b);
        CHECK(picked(0) == -1.0);
        CHECK(picked(2) == 1.0);
        auto both = mask.select(1, 0);
        STATIC_CHECK(std::is_same_v<typename decltype(both)::value_type, int>);
        CHECK(both.sum() == 1);
    }

    SECTION("reductions use the branch type, not the condition's") {
        auto mask = (a <=> 4.0).select(true, false, false);
        STATIC_CHECK(std::is_same_v<typename decltype(mask)::value_type, bool>);
        auto counted = mask.select(1, 0);
        CHECK(counted.sum() == 2);
        CHECK((a <=> 4.0).select(1.0, 0.0, 2.0).sum() == 6.0);
    }

    SECTION("ordering, all scalars") {
        auto sign = (a <=> 4.0).select(-1, 0, 1);
        CHECK(sign(0) == -1);
        CHECK(sign(1) == 1);
        CHECK(sign.sum() == 0);
    }

    SECTION("dynamic extents and materialization") {
        zipper::Array<double, zipper::dynamic_extent> d(3);
        d(0) = 2.0;
        d(1) = 4.0;
        d(2) = 6.0;
        zipper::Array<double, zipper::dynamic_extent> counts =
            (d <=> 4.0).select(0.0, 1.0, 0.0);
        CHECK(counts.extent(0) == 3);
        CHECK(counts(1) == 1.0);
        CHECK(counts.sum() == 1.0);
    }
}

TEST_CASE("array_scalar_min_max_clamp", "[array][minmax]") {
    zipper::Array<double, 4> a;
    a(0) = -2.0;
    a(1) = 0.5;
    a(2) = 3.0;
    a(3) = 1.0;

    const zipper::Array<double, 4> low = zipper::min(a, 1.0);
    CHECK(low(0) == -2.0);
    CHECK(low(2) == 1.0);
    const zipper::Array<double, 4> high = zipper::max(0.0, a);
    CHECK(high(0) == 0.0);
    CHECK(high(2) == 3.0);

    SECTION("scalar bounds") {
        const zipper::Array<double, 4> c = zipper::clamp(a, 0.0, 1.0);
        CHECK(c(0) == 0.0);
        CHECK(c(1) == 0.5);
        CHECK(c(2) == 1.0);
        CHECK(c(3) == 1.0);
    }
    SECTION("array bounds") {
        zipper::Array<double, 4> lo;
        zipper::Array<double, 4> hi;
        for (zipper::index_type i = 0; i < 4; ++i) {
            lo(i) = -1.0;
            hi(i) = static_cast<double>(i);
        }
        const zipper::Array<double, 4> c = zipper::clamp(a, lo, hi);
        CHECK(c(0) == -1.0);
        CHECK(c(1) == 0.5);
        CHECK(c(2) == 2.0);
    }
    SECTION("the expression owns its intermediates") {
        const auto make = [](const zipper::Array<double, 4> &x) {
            return zipper::clamp(x, 0.0, 1.0);
        };
        const auto lazy = make(a);
        CHECK(lazy(2) == 1.0);
        CHECK(lazy.sum() == 2.5);
    }
}

TEST_CASE("zipper_fill", "[array][fill]") {
    zipper::Array<float, 3> a;
    a.fill(2);
    CHECK(a.sum() == 6.0f);

    // A slice of a rank-3 array.
    zipper::Array<float, zipper::dynamic_extent, zipper::dynamic_extent, 4> rgba(2, 3);
    rgba.fill(0.0f);
    const zipper::full_extent_t all{};
    rgba.slice(all, all, 3).fill(1.0f);
    CHECK(rgba(1, 2, 3) == 1.0f);
    CHECK(rgba(1, 2, 2) == 0.0f);
    CHECK(rgba.sum() == 6.0f);

    zipper::Vector<double, 3> v;
    v.fill(0.5);
    CHECK(v.as_array().sum() == 1.5);
}

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

namespace {
// A (2, 3, 4) array with a(i, j, k) = 100 i + 10 j + k.
auto make_tensor() {
    zipper::Array<double, 2, 3, 4> a;
    for (zipper::index_type i = 0; i < 2; ++i) {
        for (zipper::index_type j = 0; j < 3; ++j) {
            for (zipper::index_type k = 0; k < 4; ++k) {
                a(i, j, k) = 100.0 * i + 10.0 * j + k;
            }
        }
    }
    return a;
}
} // namespace

TEST_CASE("array_over_axes", "[array][over]") {
    const auto a = make_tensor();

    SECTION("reductions over one axis") {
        const auto s = a.over<2>().sum(); // (2, 3)
        STATIC_CHECK(decltype(s)::extents_type::rank() == 2);
        CHECK(s(1, 2) == 4 * 120.0 + 6.0);
        CHECK(a.over<0>().max()(2, 3) == 123.0);
    }
    SECTION("reductions over several axes") {
        const auto s = a.over<0, 1>().sum(); // (4)
        STATIC_CHECK(decltype(s)::extents_type::rank() == 1);
        // sum_ij (100 i + 10 j + k) = 6k + 3*100 + 2*30
        CHECK(s(3) == 6 * 3.0 + 300.0 + 60.0);
    }
    SECTION("contraction with a vector") {
        zipper::Array<double, 4> w;
        w(0) = 1.0;
        w(1) = 0.0;
        w(2) = 0.0;
        w(3) = 2.0;
        const auto c = a.over<2>().contract(w); // (2, 3)
        CHECK(c(1, 1) == 110.0 + 2.0 * 113.0);
    }
    SECTION("contraction with a tensor") {
        zipper::Array<double, 3, 4> w;
        w.fill(0.0);
        w(2, 1) = 1.0;
        w(0, 3) = -1.0;
        const auto c = a.over<1, 2>().contract(w); // (2)
        CHECK(c(1) == 121.0 - 103.0);
        // A temporary weight tensor is owned by the result.
        const auto owned = a.over<1, 2>().contract(zipper::Array<double, 3, 4>(w));
        CHECK(owned(0) == 21.0 - 3.0);
    }
}

TEST_CASE("array_broadcast", "[array][broadcast]") {
    zipper::Array<double, 4> v;
    for (zipper::index_type k = 0; k < 4; ++k) v(k) = static_cast<double>(k);

    const auto b = v.broadcast<0, 1>(zipper::extents<2, 3, 4>{});
    STATIC_CHECK(decltype(b)::extents_type::rank() == 3);
    CHECK(b.extent(1) == 3);
    CHECK(b(1, 2, 3) == 3.0);
    CHECK(b(0, 0, 2) == 2.0);

    // A new middle axis.
    zipper::Array<double, 2, 4> m;
    for (zipper::index_type i = 0; i < 2; ++i) {
        for (zipper::index_type k = 0; k < 4; ++k) m(i, k) = 10.0 * i + k;
    }
    const auto mid = m.broadcast<1>(zipper::dextents<3>(2, 5, 4));
    CHECK(mid(1, 4, 3) == 13.0);

    // Per-channel scaling.
    const auto a = make_tensor();
    const zipper::Array<double, 2, 3, 4> scaled = a * zipper::as_array(v.broadcast<0, 1>(a.extents()));
    CHECK(scaled(1, 2, 3) == 123.0 * 3.0);
    CHECK(scaled(1, 2, 0) == 0.0);
}

TEST_CASE("array_over_map", "[array][over][map]") {
    // (3, 2, 2) channel-first RGB: a(c, x, y) = 10 c + 2 x + y.
    zipper::Array<double, 3, 2, 2> rgb;
    for (zipper::index_type c = 0; c < 3; ++c) {
        for (zipper::index_type x = 0; x < 2; ++x) {
            for (zipper::index_type y = 0; y < 2; ++y) rgb(c, x, y) = 10.0 * c + 2.0 * x + y;
        }
    }
    int calls = 0;
    const auto to_rgba = [&calls](const auto &c) {
        ++calls;
        return zipper::Vector<double, 4>{c(0), c(1), c(2), 1.0};
    };

    SECTION("channels stay first: (3, W, H) -> (4, W, H)") {
        const zipper::Array<double, 4, 2, 2> rgba = rgb.over<0>().map<0>(to_rgba);
        CHECK(calls == 4); // once per pixel
        CHECK(rgba(2, 1, 0) == 22.0);
        CHECK(rgba(3, 1, 1) == 1.0);
    }
    SECTION("channels move last: (3, W, H) -> (W, H, 4)") {
        const auto lazy = rgb.over<0>().map<2>(to_rgba);
        STATIC_CHECK(std::is_same_v<typename decltype(lazy)::extents_type, zipper::extents<2, 2, 4>>);
        const zipper::Array<double, 2, 2, 4> rgba = lazy;
        CHECK(rgba(1, 0, 2) == 22.0);
        CHECK(rgba(0, 1, 3) == 1.0);
    }
    SECTION("channels last, dropped: (W, H, 4) -> (W, H, 3)") {
        const zipper::Array<double, 2, 2, 4> rgba = rgb.over<0>().map<2>(to_rgba);
        const zipper::Array<double, 2, 2, 3> back = rgba.over<2>().map<2>([](const auto &c) {
            return zipper::Vector<double, 3>{c(0), c(1), c(2)};
        });
        CHECK(back(1, 1, 1) == 13.0);
    }
    SECTION("default placement is in place") {
        const zipper::Array<double, 4, 2, 2> rgba = rgb.over<0>().map(to_rgba);
        CHECK(rgba(0, 1, 1) == 3.0);
    }
    SECTION("results of another rank: (3, W, H) -> (W, 3, 3, H)") {
        // Each pixel's outer product with itself, placed at axes 1 and 2.
        const auto outer = rgb.over<0>().map<1, 2>([](const auto &c) {
            zipper::Array<double, 3, 3> m;
            for (zipper::index_type i = 0; i < 3; ++i) {
                for (zipper::index_type j = 0; j < 3; ++j) m(i, j) = c(i) * c(j);
            }
            return m;
        });
        STATIC_CHECK(std::is_same_v<typename decltype(outer)::extents_type,
                                    zipper::extents<2, 3, 3, 2>>);
        CHECK(outer(1, 2, 1, 0) == 22.0 * 12.0);
    }
}

TEST_CASE("array_over_map_dynamic_results", "[array][over][map]") {
    zipper::Array<double, zipper::dynamic_extent, zipper::dynamic_extent> a(2, 3);
    for (zipper::index_type i = 0; i < 2; ++i) {
        for (zipper::index_type j = 0; j < 3; ++j) a(i, j) = 3.0 * i + j;
    }
    int calls = 0;
    // Each row with its sum appended: (2, 3) -> (2, 4), sizes known only at
    // run time.
    const auto append_sum = [&calls](const auto &row) {
        ++calls;
        zipper::Array<double, zipper::dynamic_extent> out(row.extent(0) + 1);
        for (zipper::index_type j = 0; j < row.extent(0); ++j) out(j) = row(j);
        out(row.extent(0)) = zipper::as_array(row).sum();
        return out;
    };

    SECTION("extents from the first row") {
        const auto lazy = a.over<1>().map(append_sum);
        CHECK(calls == 1);
        CHECK(lazy.extent(1) == 4);
        const zipper::Array<double, zipper::dynamic_extent, zipper::dynamic_extent> result = lazy;
        CHECK(calls == 3);
        CHECK(result(1, 3) == 3.0 + 4.0 + 5.0);
    }
    SECTION("extents given") {
        const auto lazy = a.over<1>().map(append_sum, zipper::dextents<1>(4));
        CHECK(calls == 0);
        CHECK(lazy.extent(1) == 4);
    }
}
