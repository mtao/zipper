#include <iostream>
#include <zipper/expression/nullary/MDArray.hpp>
#include <zipper/expression/nullary/StlMDArray.hpp>
#include <zipper/expression/unary/ScalarArithmetic.hpp>

#include "../../catch_include.hpp"
#include "../../fmt_include.hpp"

TEST_CASE("test_vector_scalar", "[vector][unary][scalar_arithmetic]") {
  zipper::expression::nullary::MDArray<double, zipper::extents<3>,
                                       zipper::storage::layout_left,
                                       zipper::default_accessor_policy<double>>
      x;
  x(0) = 0.5;
  x(1) = 1.5;
  x(2) = 2.5;

  {
    auto y = 3 * x;
    CHECK(y(0) == 1.5);
    CHECK(y(1) == 4.5);
    CHECK(y(2) == 7.5);
  }
  {
    auto y = x.as_array() + 1;
    CHECK(y(0) == 1.5);
    CHECK(y(1) == 2.5);
    CHECK(y(2) == 3.5);
  }
}
TEST_CASE("test_matrix_scalar", "[matrix][unary][scalar_arithmetic]") {
  zipper::Matrix<double, 1, 3> x{{0.5, 1.5, 2.5}};

  {
    auto y = 3 * x;
    CHECK(y(0, 0) == 1.5);
    CHECK(y(0, 1) == 4.5);
    CHECK(y(0, 2) == 7.5);
  }
  {
    auto y = x.as_array() + 1;
    CHECK(y(0, 0) == 1.5);
    CHECK(y(0, 1) == 2.5);
    CHECK(y(0, 2) == 3.5);
  }
}
