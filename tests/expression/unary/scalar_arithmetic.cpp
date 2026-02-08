#include <iostream>
#include <zipper/expression/nullary/MDArray.hpp>
#include <zipper/expression/nullary/StlMDArray.hpp>
#include <zipper/expression/unary/ScalarArithmetic.hpp>

#include "../../catch_include.hpp"

TEST_CASE("test_vector_scalar_multiplies", "[vector][unary][scalar_arithmetic]") {
  zipper::expression::nullary::MDArray<double, zipper::extents<3>,
                                       zipper::storage::layout_left,
                                       zipper::default_accessor_policy<double>>
      x;
  x(0) = 0.5;
  x(1) = 1.5;
  x(2) = 2.5;

  {
    // Test scalar * expression (scalar on left)
    using expr_type = std::decay_t<decltype(x)>;
    zipper::expression::unary::ScalarMultiplies<double, const expr_type, false> y(3.0, x);
    CHECK(y(0) == 1.5);
    CHECK(y(1) == 4.5);
    CHECK(y(2) == 7.5);
  }
  {
    // Test expression * scalar (scalar on right)
    using expr_type = std::decay_t<decltype(x)>;
    zipper::expression::unary::ScalarMultiplies<double, const expr_type, true> y(x, 2.0);
    CHECK(y(0) == 1.0);
    CHECK(y(1) == 3.0);
    CHECK(y(2) == 5.0);
  }
  {
    // Test scalar + expression
    using expr_type = std::decay_t<decltype(x)>;
    zipper::expression::unary::ScalarPlus<double, const expr_type, false> y(1.0, x);
    CHECK(y(0) == 1.5);
    CHECK(y(1) == 2.5);
    CHECK(y(2) == 3.5);
  }
  {
    // Test scalar - expression
    using expr_type = std::decay_t<decltype(x)>;
    zipper::expression::unary::ScalarMinus<double, const expr_type, false> y(10.0, x);
    CHECK(y(0) == 9.5);
    CHECK(y(1) == 8.5);
    CHECK(y(2) == 7.5);
  }
  {
    // Test negate expression
    using expr_type = std::decay_t<decltype(x)>;
    zipper::expression::unary::Negate<const expr_type> y(x);
    CHECK(y(0) == -0.5);
    CHECK(y(1) == -1.5);
    CHECK(y(2) == -2.5);
  }
  {
    // Test scalar divides (expression / scalar)
    using expr_type = std::decay_t<decltype(x)>;
    zipper::expression::unary::ScalarDivides<double, const expr_type, true> y(x, 2.0);
    CHECK(y(0) == 0.25);
    CHECK(y(1) == 0.75);
    CHECK(y(2) == 1.25);
  }
}

TEST_CASE("test_matrix_scalar_multiplies", "[matrix][unary][scalar_arithmetic]") {
  zipper::expression::nullary::MDArray<double, zipper::extents<1, 3>,
                                       zipper::storage::layout_left,
                                       zipper::default_accessor_policy<double>>
      x;
  x(0, 0) = 0.5;
  x(0, 1) = 1.5;
  x(0, 2) = 2.5;

  {
    using expr_type = std::decay_t<decltype(x)>;
    zipper::expression::unary::ScalarMultiplies<double, const expr_type, false> y(3.0, x);
    CHECK(y(0, 0) == 1.5);
    CHECK(y(0, 1) == 4.5);
    CHECK(y(0, 2) == 7.5);
  }
  {
    using expr_type = std::decay_t<decltype(x)>;
    zipper::expression::unary::ScalarPlus<double, const expr_type, false> y(1.0, x);
    CHECK(y(0, 0) == 1.5);
    CHECK(y(0, 1) == 2.5);
    CHECK(y(0, 2) == 3.5);
  }
}
