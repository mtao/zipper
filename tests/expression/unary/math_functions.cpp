// Tests for ArrayBase math function member functions
// (sqrt, cbrt, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
//  exp, exp2, log, log2, log10, floor, ceil, round, sign)

#include <zipper/Array.hpp>
#include <zipper/Vector.hpp>
#include <zipper/ArrayBase.hxx>

#include "../../catch_include.hpp"

#include <cmath>
#include <numbers>

// ──────────────────── sqrt ────────────────────

TEST_CASE("sqrt_basic", "[sqrt][unary]") {
    zipper::Array<double, 3> x;
    x(0) = 0.0;
    x(1) = 1.0;
    x(2) = 4.0;

    auto r = x.sqrt();
    CHECK(r(0) == Catch::Approx(0.0));
    CHECK(r(1) == Catch::Approx(1.0));
    CHECK(r(2) == Catch::Approx(2.0));
}

TEST_CASE("sqrt_is_lazy", "[sqrt][unary][view]") {
    zipper::Array<double, 2> x;
    x(0) = 9.0;
    x(1) = 16.0;

    auto r = x.sqrt();
    CHECK(r(0) == Catch::Approx(3.0));

    x(0) = 25.0;
    CHECK(r(0) == Catch::Approx(5.0));
}

// ──────────────────── cbrt ────────────────────

TEST_CASE("cbrt_basic", "[cbrt][unary]") {
    zipper::Array<double, 3> x;
    x(0) = 0.0;
    x(1) = 8.0;
    x(2) = -27.0;

    auto r = x.cbrt();
    CHECK(r(0) == Catch::Approx(0.0));
    CHECK(r(1) == Catch::Approx(2.0));
    CHECK(r(2) == Catch::Approx(-3.0));
}

// ──────────────────── sin ────────────────────

TEST_CASE("sin_basic", "[sin][unary]") {
    zipper::Array<double, 3> x;
    x(0) = 0.0;
    x(1) = std::numbers::pi / 2.0;
    x(2) = std::numbers::pi;

    auto r = x.sin();
    CHECK(r(0) == Catch::Approx(0.0));
    CHECK(r(1) == Catch::Approx(1.0));
    CHECK(r(2) == Catch::Approx(0.0).margin(1e-15));
}

// ──────────────────── cos ────────────────────

TEST_CASE("cos_basic", "[cos][unary]") {
    zipper::Array<double, 3> x;
    x(0) = 0.0;
    x(1) = std::numbers::pi / 2.0;
    x(2) = std::numbers::pi;

    auto r = x.cos();
    CHECK(r(0) == Catch::Approx(1.0));
    CHECK(r(1) == Catch::Approx(0.0).margin(1e-15));
    CHECK(r(2) == Catch::Approx(-1.0));
}

// ──────────────────── tan ────────────────────

TEST_CASE("tan_basic", "[tan][unary]") {
    zipper::Array<double, 3> x;
    x(0) = 0.0;
    x(1) = std::numbers::pi / 4.0;
    x(2) = -std::numbers::pi / 4.0;

    auto r = x.tan();
    CHECK(r(0) == Catch::Approx(0.0));
    CHECK(r(1) == Catch::Approx(1.0));
    CHECK(r(2) == Catch::Approx(-1.0));
}

// ──────────────────── asin ────────────────────

TEST_CASE("asin_basic", "[asin][unary]") {
    zipper::Array<double, 3> x;
    x(0) = 0.0;
    x(1) = 1.0;
    x(2) = -1.0;

    auto r = x.asin();
    CHECK(r(0) == Catch::Approx(0.0));
    CHECK(r(1) == Catch::Approx(std::numbers::pi / 2.0));
    CHECK(r(2) == Catch::Approx(-std::numbers::pi / 2.0));
}

// ──────────────────── acos ────────────────────

TEST_CASE("acos_basic", "[acos][unary]") {
    zipper::Array<double, 3> x;
    x(0) = 1.0;
    x(1) = 0.0;
    x(2) = -1.0;

    auto r = x.acos();
    CHECK(r(0) == Catch::Approx(0.0));
    CHECK(r(1) == Catch::Approx(std::numbers::pi / 2.0));
    CHECK(r(2) == Catch::Approx(std::numbers::pi));
}

// ──────────────────── atan ────────────────────

TEST_CASE("atan_basic", "[atan][unary]") {
    zipper::Array<double, 3> x;
    x(0) = 0.0;
    x(1) = 1.0;
    x(2) = -1.0;

    auto r = x.atan();
    CHECK(r(0) == Catch::Approx(0.0));
    CHECK(r(1) == Catch::Approx(std::numbers::pi / 4.0));
    CHECK(r(2) == Catch::Approx(-std::numbers::pi / 4.0));
}

// ──────────────────── sinh ────────────────────

TEST_CASE("sinh_basic", "[sinh][unary]") {
    zipper::Array<double, 3> x;
    x(0) = 0.0;
    x(1) = 1.0;
    x(2) = -1.0;

    auto r = x.sinh();
    CHECK(r(0) == Catch::Approx(0.0));
    CHECK(r(1) == Catch::Approx(std::sinh(1.0)));
    CHECK(r(2) == Catch::Approx(std::sinh(-1.0)));
}

// ──────────────────── cosh ────────────────────

TEST_CASE("cosh_basic", "[cosh][unary]") {
    zipper::Array<double, 3> x;
    x(0) = 0.0;
    x(1) = 1.0;
    x(2) = -1.0;

    auto r = x.cosh();
    CHECK(r(0) == Catch::Approx(1.0));  // cosh(0) = 1, NOT zero-preserving
    CHECK(r(1) == Catch::Approx(std::cosh(1.0)));
    CHECK(r(2) == Catch::Approx(std::cosh(-1.0)));
}

// ──────────────────── tanh ────────────────────

TEST_CASE("tanh_basic", "[tanh][unary]") {
    zipper::Array<double, 3> x;
    x(0) = 0.0;
    x(1) = 1.0;
    x(2) = -1.0;

    auto r = x.tanh();
    CHECK(r(0) == Catch::Approx(0.0));
    CHECK(r(1) == Catch::Approx(std::tanh(1.0)));
    CHECK(r(2) == Catch::Approx(std::tanh(-1.0)));
}

// ──────────────────── exp ────────────────────

TEST_CASE("exp_basic", "[exp][unary]") {
    zipper::Array<double, 3> x;
    x(0) = 0.0;
    x(1) = 1.0;
    x(2) = 2.0;

    auto r = x.exp();
    CHECK(r(0) == Catch::Approx(1.0));  // exp(0) = 1, NOT zero-preserving
    CHECK(r(1) == Catch::Approx(std::numbers::e));
    CHECK(r(2) == Catch::Approx(std::exp(2.0)));
}

// ──────────────────── exp2 ────────────────────

TEST_CASE("exp2_basic", "[exp2][unary]") {
    zipper::Array<double, 3> x;
    x(0) = 0.0;
    x(1) = 1.0;
    x(2) = 3.0;

    auto r = x.exp2();
    CHECK(r(0) == Catch::Approx(1.0));  // 2^0 = 1, NOT zero-preserving
    CHECK(r(1) == Catch::Approx(2.0));
    CHECK(r(2) == Catch::Approx(8.0));
}

// ──────────────────── log ────────────────────

TEST_CASE("log_basic", "[log][unary]") {
    zipper::Array<double, 3> x;
    x(0) = 1.0;
    x(1) = std::numbers::e;
    x(2) = std::exp(2.0);

    auto r = x.log();
    CHECK(r(0) == Catch::Approx(0.0));
    CHECK(r(1) == Catch::Approx(1.0));
    CHECK(r(2) == Catch::Approx(2.0));
}

// ──────────────────── log2 ────────────────────

TEST_CASE("log2_basic", "[log2][unary]") {
    zipper::Array<double, 3> x;
    x(0) = 1.0;
    x(1) = 2.0;
    x(2) = 8.0;

    auto r = x.log2();
    CHECK(r(0) == Catch::Approx(0.0));
    CHECK(r(1) == Catch::Approx(1.0));
    CHECK(r(2) == Catch::Approx(3.0));
}

// ──────────────────── log10 ────────────────────

TEST_CASE("log10_basic", "[log10][unary]") {
    zipper::Array<double, 3> x;
    x(0) = 1.0;
    x(1) = 10.0;
    x(2) = 1000.0;

    auto r = x.log10();
    CHECK(r(0) == Catch::Approx(0.0));
    CHECK(r(1) == Catch::Approx(1.0));
    CHECK(r(2) == Catch::Approx(3.0));
}

// ──────────────────── floor ────────────────────

TEST_CASE("floor_basic", "[floor][unary]") {
    zipper::Array<double, 4> x;
    x(0) = 0.0;
    x(1) = 2.7;
    x(2) = -2.3;
    x(3) = 3.0;

    auto r = x.floor();
    CHECK(r(0) == Catch::Approx(0.0));
    CHECK(r(1) == Catch::Approx(2.0));
    CHECK(r(2) == Catch::Approx(-3.0));
    CHECK(r(3) == Catch::Approx(3.0));
}

// ──────────────────── ceil ────────────────────

TEST_CASE("ceil_basic", "[ceil][unary]") {
    zipper::Array<double, 4> x;
    x(0) = 0.0;
    x(1) = 2.3;
    x(2) = -2.7;
    x(3) = 3.0;

    auto r = x.ceil();
    CHECK(r(0) == Catch::Approx(0.0));
    CHECK(r(1) == Catch::Approx(3.0));
    CHECK(r(2) == Catch::Approx(-2.0));
    CHECK(r(3) == Catch::Approx(3.0));
}

// ──────────────────── round ────────────────────

TEST_CASE("round_basic", "[round][unary]") {
    zipper::Array<double, 4> x;
    x(0) = 0.0;
    x(1) = 2.3;
    x(2) = 2.7;
    x(3) = -2.5;

    auto r = x.round();
    CHECK(r(0) == Catch::Approx(0.0));
    CHECK(r(1) == Catch::Approx(2.0));
    CHECK(r(2) == Catch::Approx(3.0));
    CHECK(r(3) == Catch::Approx(std::round(-2.5)));
}

// ──────────────────── sign ────────────────────

TEST_CASE("sign_basic", "[sign][unary]") {
    zipper::Array<double, 4> x;
    x(0) = 5.0;
    x(1) = -3.0;
    x(2) = 0.0;
    x(3) = -0.0;

    auto r = x.sign();
    CHECK(r(0) == Catch::Approx(1.0));
    CHECK(r(1) == Catch::Approx(-1.0));
    CHECK(r(2) == Catch::Approx(0.0));
    CHECK(r(3) == Catch::Approx(0.0));
}

// ──────────────────── composition / chaining ────────────────────

TEST_CASE("chained_math_functions", "[math][unary][composition]") {
    // sqrt(abs(x)) should work via chaining
    zipper::Array<double, 3> x;
    x(0) = -4.0;
    x(1) = -9.0;
    x(2) = -16.0;

    auto r = x.abs().sqrt();
    CHECK(r(0) == Catch::Approx(2.0));
    CHECK(r(1) == Catch::Approx(3.0));
    CHECK(r(2) == Catch::Approx(4.0));
}

TEST_CASE("exp_log_roundtrip", "[math][unary][composition]") {
    zipper::Array<double, 3> x;
    x(0) = 1.0;
    x(1) = 2.5;
    x(2) = 0.1;

    // log(exp(x)) == x
    auto r = x.exp().log();
    CHECK(r(0) == Catch::Approx(1.0));
    CHECK(r(1) == Catch::Approx(2.5));
    CHECK(r(2) == Catch::Approx(0.1));
}

TEST_CASE("sin_asin_roundtrip", "[math][unary][composition]") {
    zipper::Array<double, 3> x;
    x(0) = 0.0;
    x(1) = 0.5;
    x(2) = -0.7;

    // asin(sin(x)) == x for x in [-pi/2, pi/2]
    auto r = x.sin().asin();
    CHECK(r(0) == Catch::Approx(0.0).margin(1e-15));
    CHECK(r(1) == Catch::Approx(0.5));
    CHECK(r(2) == Catch::Approx(-0.7));
}

// ──────────────────── 2D array ────────────────────

TEST_CASE("math_functions_2d", "[math][unary]") {
    zipper::Array<double, 2, 2> x;
    x(0, 0) = 0.0;
    x(0, 1) = 1.0;
    x(1, 0) = 4.0;
    x(1, 1) = 9.0;

    auto r = x.sqrt();
    CHECK(r(0, 0) == Catch::Approx(0.0));
    CHECK(r(0, 1) == Catch::Approx(1.0));
    CHECK(r(1, 0) == Catch::Approx(2.0));
    CHECK(r(1, 1) == Catch::Approx(3.0));
}

// ──────────────────── eval materializes ────────────────────

TEST_CASE("math_function_eval", "[math][unary][eval]") {
    zipper::Array<double, 3> x;
    x(0) = 1.0;
    x(1) = 4.0;
    x(2) = 9.0;

    // eval() materializes the lazy expression into an owned Array
    auto r = x.sqrt().eval();
    CHECK(r(0) == Catch::Approx(1.0));
    CHECK(r(1) == Catch::Approx(2.0));
    CHECK(r(2) == Catch::Approx(3.0));

    // Changing x should NOT affect r (it's materialized)
    x(0) = 100.0;
    CHECK(r(0) == Catch::Approx(1.0));
}
