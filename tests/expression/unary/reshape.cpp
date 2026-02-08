#include "../../catch_include.hpp"
#include <zipper/Vector.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/expression/unary/Reshape.hpp>

using namespace zipper;

TEST_CASE("test_reshape_1d_to_2d", "[expression][unary][reshape]") {
  // Reshape a 6-element vector into a 2x3 matrix
  Vector<double, 6> v{1, 2, 3, 4, 5, 6};

  using new_extents = zipper::extents<2, 3>;
  auto r = expression::unary::Reshape(v.expression(), new_extents{});

  // layout_right (row-major):
  // v[0]=1 v[1]=2 v[2]=3 v[3]=4 v[4]=5 v[5]=6
  // becomes:
  // [[1, 2, 3],
  //  [4, 5, 6]]
  CHECK(r(0, 0) == 1);
  CHECK(r(0, 1) == 2);
  CHECK(r(0, 2) == 3);
  CHECK(r(1, 0) == 4);
  CHECK(r(1, 1) == 5);
  CHECK(r(1, 2) == 6);
}

TEST_CASE("test_reshape_2d_to_1d", "[expression][unary][reshape]") {
  // Reshape a 2x3 matrix into a 6-element vector
  Matrix<double, 2, 3> m{{1, 2, 3}, {4, 5, 6}};

  using new_extents = zipper::extents<6>;
  auto r = expression::unary::Reshape(m.expression(), new_extents{});

  CHECK(r(0) == 1);
  CHECK(r(1) == 2);
  CHECK(r(2) == 3);
  CHECK(r(3) == 4);
  CHECK(r(4) == 5);
  CHECK(r(5) == 6);
}

TEST_CASE("test_reshape_2d_to_2d", "[expression][unary][reshape]") {
  // Reshape a 2x3 matrix into a 3x2 matrix (not transpose!)
  Matrix<double, 2, 3> m{{1, 2, 3}, {4, 5, 6}};

  using new_extents = zipper::extents<3, 2>;
  auto r = expression::unary::Reshape(m.expression(), new_extents{});

  // Row-major reinterpretation:
  // [1,2,3,4,5,6] → [[1,2],[3,4],[5,6]]
  CHECK(r(0, 0) == 1);
  CHECK(r(0, 1) == 2);
  CHECK(r(1, 0) == 3);
  CHECK(r(1, 1) == 4);
  CHECK(r(2, 0) == 5);
  CHECK(r(2, 1) == 6);
}

TEST_CASE("test_reshape_static_construct", "[expression][unary][reshape]") {
  // Test the static-extents constructor (no extents argument needed)
  Vector<double, 4> v{10, 20, 30, 40};

  using new_extents = zipper::extents<2, 2>;
  auto r = expression::unary::Reshape<
      std::decay_t<decltype(v.expression())>, new_extents>(v.expression());

  CHECK(r(0, 0) == 10);
  CHECK(r(0, 1) == 20);
  CHECK(r(1, 0) == 30);
  CHECK(r(1, 1) == 40);
}
