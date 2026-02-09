// Tests for LpNorm and LpNormPowered reductions

#include <cmath>
#include <zipper/Array.hpp>
#include <zipper/Vector.hpp>
#include <zipper/ArrayBase.hxx>

#include "../../catch_include.hpp"

// ============================================================
// VectorBase::norm() — L2 norm by default
// ============================================================

TEST_CASE("vector_l2_norm", "[reduction][norm]") {
    zipper::Vector<double, 3> x{3.0, 4.0, 0.0};
    CHECK(x.norm() == Catch::Approx(5.0));
}

TEST_CASE("vector_l2_norm_unit", "[reduction][norm]") {
    zipper::Vector<double, 3> x{1.0, 0.0, 0.0};
    CHECK(x.norm() == Catch::Approx(1.0));
}

TEST_CASE("vector_l2_norm_general", "[reduction][norm]") {
    zipper::Vector<double, 3> x{1.0, 2.0, 3.0};
    CHECK(x.norm() == Catch::Approx(std::sqrt(14.0)));
}

// ============================================================
// VectorBase::norm<1>() — L1 norm
// ============================================================

TEST_CASE("vector_l1_norm", "[reduction][norm]") {
    zipper::Vector<double, 3> x{-1.0, 2.0, -3.0};
    CHECK(x.norm<1>() == Catch::Approx(6.0));
}

TEST_CASE("vector_l1_norm_positive", "[reduction][norm]") {
    zipper::Vector<double, 3> x{1.0, 2.0, 3.0};
    CHECK(x.norm<1>() == Catch::Approx(6.0));
}

// ============================================================
// VectorBase::norm_powered — squared norm etc.
// ============================================================

TEST_CASE("vector_squared_norm", "[reduction][norm]") {
    zipper::Vector<double, 3> x{3.0, 4.0, 0.0};
    CHECK(x.norm_powered<2>() == Catch::Approx(25.0));
}

TEST_CASE("vector_squared_norm_general", "[reduction][norm]") {
    zipper::Vector<double, 3> x{1.0, 2.0, 3.0};
    CHECK(x.norm_powered<2>() == Catch::Approx(14.0));
}

// ============================================================
// ArrayBase::norm() — works the same way
// ============================================================

TEST_CASE("array_l2_norm", "[reduction][norm][array]") {
    zipper::Array<double, 3> x;
    x(0) = 3.0;
    x(1) = 4.0;
    x(2) = 0.0;
    CHECK(x.norm() == Catch::Approx(5.0));
}

TEST_CASE("array_l1_norm", "[reduction][norm][array]") {
    zipper::Array<double, 3> x;
    x(0) = -1.0;
    x(1) = 2.0;
    x(2) = -3.0;
    CHECK(x.norm<1>() == Catch::Approx(6.0));
}

// ============================================================
// 2D array norms
// ============================================================

TEST_CASE("array_2d_norm", "[reduction][norm][array]") {
    zipper::Array<double, 2, 2> x;
    x(0, 0) = 1.0;
    x(0, 1) = 2.0;
    x(1, 0) = 3.0;
    x(1, 1) = 4.0;
    // Frobenius norm = sqrt(1+4+9+16) = sqrt(30)
    CHECK(x.norm() == Catch::Approx(std::sqrt(30.0)));
}

// ============================================================
// normalized() — returns unit vector
// ============================================================

TEST_CASE("vector_normalized", "[reduction][norm]") {
    zipper::Vector<double, 3> x{3.0, 4.0, 0.0};
    auto n = x.normalized();
    CHECK(n(0) == Catch::Approx(0.6));
    CHECK(n(1) == Catch::Approx(0.8));
    CHECK(n(2) == Catch::Approx(0.0));
}

// ============================================================
// Dynamic vector norms
// ============================================================

TEST_CASE("dynamic_vector_l2_norm", "[reduction][norm][dynamic]") {
    zipper::Vector<double, std::dynamic_extent> x = {3.0, 4.0};
    CHECK(x.norm() == Catch::Approx(5.0));
}
