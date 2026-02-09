// Tests for coefficient reductions: Sum, Product, All, Any

#include <zipper/Array.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/ArrayBase.hxx>

#include "../../catch_include.hpp"

// ============================================================
// CoefficientSum
// ============================================================

TEST_CASE("coefficient_sum_vector", "[reduction][sum]") {
    zipper::Vector<double, 4> x{1.0, 2.0, 3.0, 4.0};
    auto xa = x.as_array();
    CHECK(xa.sum() == 10.0);
}

TEST_CASE("coefficient_sum_matrix", "[reduction][sum]") {
    zipper::Matrix<double, 2, 2> M{{1.0, 2.0}, {3.0, 4.0}};
    auto Ma = M.as_array();
    CHECK(Ma.sum() == 10.0);
}

TEST_CASE("coefficient_sum_zeros", "[reduction][sum]") {
    zipper::Array<double, 3> x;
    x(0) = 0.0;
    x(1) = 0.0;
    x(2) = 0.0;
    CHECK(x.sum() == 0.0);
}

TEST_CASE("coefficient_sum_negative", "[reduction][sum]") {
    zipper::Array<double, 3> x;
    x(0) = -1.0;
    x(1) = 2.0;
    x(2) = -3.0;
    CHECK(x.sum() == -2.0);
}

// ============================================================
// CoefficientProduct
// ============================================================

TEST_CASE("coefficient_product_vector", "[reduction][product]") {
    zipper::Vector<double, 3> x{1.0, 2.0, 3.0};
    auto xa = x.as_array();
    CHECK(xa.product() == 6.0);
}

TEST_CASE("coefficient_product_matrix", "[reduction][product]") {
    zipper::Matrix<double, 2, 2> M{{1.0, 2.0}, {3.0, 4.0}};
    auto Ma = M.as_array();
    CHECK(Ma.product() == 24.0);
}

TEST_CASE("coefficient_product_with_zero", "[reduction][product]") {
    zipper::Array<double, 3> x;
    x(0) = 5.0;
    x(1) = 0.0;
    x(2) = 3.0;
    CHECK(x.product() == 0.0);
}

TEST_CASE("coefficient_product_negative", "[reduction][product]") {
    zipper::Array<double, 3> x;
    x(0) = -1.0;
    x(1) = 2.0;
    x(2) = -3.0;
    CHECK(x.product() == 6.0);
}

// ============================================================
// All
// ============================================================

TEST_CASE("all_true", "[reduction][all]") {
    zipper::Array<double, 3> a;
    a(0) = 1.0;
    a(1) = 2.0;
    a(2) = 3.0;
    zipper::Array<double, 3> b;
    b(0) = 1.0;
    b(1) = 2.0;
    b(2) = 3.0;

    auto cmp = (a == b);
    CHECK(cmp.all());
}

TEST_CASE("all_false", "[reduction][all]") {
    zipper::Array<double, 3> a;
    a(0) = 1.0;
    a(1) = 2.0;
    a(2) = 3.0;
    zipper::Array<double, 3> b;
    b(0) = 1.0;
    b(1) = 99.0;
    b(2) = 3.0;

    auto cmp = (a == b);
    CHECK_FALSE(cmp.all());
}

// ============================================================
// Any
// ============================================================

TEST_CASE("any_true", "[reduction][any]") {
    zipper::Array<double, 3> a;
    a(0) = 1.0;
    a(1) = 2.0;
    a(2) = 3.0;
    zipper::Array<double, 3> b;
    b(0) = 99.0;
    b(1) = 2.0;
    b(2) = 99.0;

    auto cmp = (a == b);
    CHECK(cmp.any());
}

TEST_CASE("any_false", "[reduction][any]") {
    zipper::Array<double, 3> a;
    a(0) = 1.0;
    a(1) = 2.0;
    a(2) = 3.0;
    zipper::Array<double, 3> b;
    b(0) = 10.0;
    b(1) = 20.0;
    b(2) = 30.0;

    auto cmp = (a == b);
    CHECK_FALSE(cmp.any());
}

TEST_CASE("any_all_interaction", "[reduction][any][all]") {
    zipper::Array<double, 3> a;
    a(0) = 1.0;
    a(1) = 2.0;
    a(2) = 3.0;

    // All greater than 0
    auto gt = (a > 0.0);
    CHECK(gt.all());
    CHECK(gt.any());

    // None greater than 10
    auto gt10 = (a > 10.0);
    CHECK_FALSE(gt10.all());
    CHECK_FALSE(gt10.any());

    // Some greater than 1
    auto gt1 = (a > 1.0);
    CHECK_FALSE(gt1.all());
    CHECK(gt1.any());
}
