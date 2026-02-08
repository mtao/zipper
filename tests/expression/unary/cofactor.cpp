#include "../../catch_include.hpp"
#include <zipper/Vector.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/expression/unary/Cofactor.hpp>

using namespace zipper;

TEST_CASE("test_cofactor_2x2", "[expression][unary][cofactor]") {
  // M = [[a, b], [c, d]]
  // Cofactor: C(0,0) = d, C(0,1) = -c, C(1,0) = -b, C(1,1) = a
  Matrix<double, 2, 2> m{{3, 8}, {4, 6}};

  auto c = expression::unary::Cofactor(m.expression());

  CHECK(c(0, 0) == Catch::Approx(6));   // +det([6]) = 6
  CHECK(c(0, 1) == Catch::Approx(-4));  // -det([4]) = -4
  CHECK(c(1, 0) == Catch::Approx(-8));  // -det([8]) = -8
  CHECK(c(1, 1) == Catch::Approx(3));   // +det([3]) = 3
}

TEST_CASE("test_cofactor_3x3", "[expression][unary][cofactor]") {
  // M = [[1, 2, 3],
  //      [0, 4, 5],
  //      [1, 0, 6]]
  Matrix<double, 3, 3> m{{1, 2, 3}, {0, 4, 5}, {1, 0, 6}};

  auto c = expression::unary::Cofactor(m.expression());

  // C(0,0) = +det([[4,5],[0,6]]) = +(24-0) = 24
  CHECK(c(0, 0) == Catch::Approx(24));
  // C(0,1) = -det([[0,5],[1,6]]) = -(0-5) = 5
  CHECK(c(0, 1) == Catch::Approx(5));
  // C(0,2) = +det([[0,4],[1,0]]) = +(0-4) = -4
  CHECK(c(0, 2) == Catch::Approx(-4));
  // C(1,0) = -det([[2,3],[0,6]]) = -(12-0) = -12
  CHECK(c(1, 0) == Catch::Approx(-12));
  // C(1,1) = +det([[1,3],[1,6]]) = +(6-3) = 3
  CHECK(c(1, 1) == Catch::Approx(3));
  // C(1,2) = -det([[1,2],[1,0]]) = -(0-2) = 2
  CHECK(c(1, 2) == Catch::Approx(2));
  // C(2,0) = +det([[2,3],[4,5]]) = +(10-12) = -2
  CHECK(c(2, 0) == Catch::Approx(-2));
  // C(2,1) = -det([[1,3],[0,5]]) = -(5-0) = -5
  CHECK(c(2, 1) == Catch::Approx(-5));
  // C(2,2) = +det([[1,2],[0,4]]) = +(4-0) = 4
  CHECK(c(2, 2) == Catch::Approx(4));
}

TEST_CASE("test_cofactor_1x1", "[expression][unary][cofactor]") {
  Matrix<double, 1, 1> m{{7}};

  auto c = expression::unary::Cofactor(m.expression());

  // Cofactor of 1x1 is just the sign: (-1)^(0+0) * 1 = 1
  CHECK(c(0, 0) == Catch::Approx(1));
}
