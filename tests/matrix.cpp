

#include <iostream>
#include <zipper/ArrayBase.hxx>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Random.hpp>

#include "catch_include.hpp"

namespace {
void print(zipper::concepts::Matrix auto const &M) {
  for (zipper::index_type j = 0; j < M.extent(0); ++j) {
    for (zipper::index_type k = 0; k < M.extent(1); ++k) {
      std::cout << M(j, k) << " ";
    }
    std::cout << std::endl;
  }
}
} // namespace

TEST_CASE("test_assignment", "[matrix][storage][dense]") {
  zipper::Matrix<double, 3, std::dynamic_extent> M(3);
  zipper::Matrix<double, 3, 5> N;
  zipper::Matrix<double, std::dynamic_extent, std::dynamic_extent> O(2, 2);
  zipper::Vector<double, 3> x;

  O(0, 0) = 0;
  O(1, 0) = 1;
  O(0, 1) = 3;
  O(1, 1) = 4;

  CHECK(O(0, 0) == 0);
  CHECK(O(1, 0) == 1);
  CHECK(O(0, 1) == 3);
  CHECK(O(1, 1) == 4);

  x(0) = 2;
  x(1) = 5;
  x(2) = 9;
  M(0, 0) = 0;
  M(1, 0) = 1;
  M(2, 0) = 2;
  M(0, 1) = 3;
  M(1, 1) = 4;
  M(2, 1) = 5;
  M(0, 2) = 6;
  M(1, 2) = 7;
  M(2, 2) = 8;

  N(0, 0) = 0;
  N(1, 0) = 1;
  N(2, 0) = 2;
  N(0, 1) = 3;
  N(1, 1) = 4;
  N(2, 1) = 5;
  N(0, 2) = 6;
  N(1, 2) = 7;
  N(2, 2) = 8;
  N(0, 3) = 9;
  N(1, 3) = 10;
  N(2, 3) = 11;
  N(0, 4) = 12;
  N(1, 4) = 13;
  N(2, 4) = 14;

  CHECK(x(0) == 2);
  CHECK(x(1) == 5);
  CHECK(x(2) == 9);
  CHECK(M(0, 0) == 0);
  CHECK(M(1, 0) == 1);
  CHECK(M(2, 0) == 2);
  CHECK(M(0, 1) == 3);
  CHECK(M(1, 1) == 4);
  CHECK(M(2, 1) == 5);
  CHECK(M(0, 2) == 6);
  CHECK(M(1, 2) == 7);
  CHECK(M(2, 2) == 8);
  CHECK(N(0, 0) == 0);
  CHECK(N(1, 0) == 1);
  CHECK(N(2, 0) == 2);
  CHECK(N(0, 1) == 3);
  CHECK(N(1, 1) == 4);
  CHECK(N(2, 1) == 5);
  CHECK(N(0, 2) == 6);
  CHECK(N(1, 2) == 7);
  CHECK(N(2, 2) == 8);
  CHECK(N(0, 3) == 9);
  CHECK(N(1, 3) == 10);
  CHECK(N(2, 3) == 11);
  CHECK(N(0, 4) == 12);
  CHECK(N(1, 4) == 13);
  CHECK(N(2, 4) == 14);
}

TEST_CASE("test_matrix_eval", "[matrix][storage][dense]") {
  zipper::Matrix<double, 3, 5> N =
      zipper::expression::nullary::uniform_random<double>({});

  auto x = (N * 2).eval();
  auto v = N.as_array();
  static_assert(std::is_same_v<std::decay_t<decltype(v)>::extents_type,
                                decltype(N)::extents_type>);
  static_assert(std::is_same_v<decltype(v.extents()), decltype(N.extents())>);
  static_assert(std::decay_t<decltype(v)>::extents_type::rank() == 2);
  // auto y = N.as_array().eval();
  print(x);
}

TEST_CASE("test_span", "[matrix][storage][dense]") {
  zipper::Matrix<zipper::index_type, 3, 6> C;
  zipper::Matrix<zipper::index_type, 3, std::dynamic_extent> RC(6);
  zipper::Matrix<zipper::index_type, std::dynamic_extent, std::dynamic_extent>
      R(3, 6);

  auto CS = C.as_span();
  auto RCS = RC.as_span();
  auto RS = R.as_span();

  REQUIRE(CS.extents() == C.extents());
  REQUIRE(RS.extents() == R.extents());
  REQUIRE(RCS.extents() == RC.extents());

  for (zipper::index_type j = 0; j < 3; ++j) {
    for (zipper::index_type k = 0; k < 6; ++k) {
      CHECK(&C(j, k) == &CS(j, k));
      CHECK(&RC(j, k) == &RCS(j, k));
      CHECK(&R(j, k) == &RS(j, k));
    }
  }
}

TEST_CASE("test_matrix_span", "[matrix][storage][dense][span]") {
  std::vector<int> vec = {0, 1, 2, 3};
  zipper::Matrix<int, 2, 2>::span_type M = std::span<int, 4>(vec);
  zipper::Matrix<int, std::dynamic_extent, std::dynamic_extent>::span_type Md(
      std::span<int>(vec), zipper::create_dextents(2, 2));

  static_assert(M.static_extent(0) == 2);
  static_assert(M.static_extent(1) == 2);
  static_assert(Md.static_extent(0) == std::dynamic_extent);
  static_assert(Md.static_extent(1) == std::dynamic_extent);
  REQUIRE(M.extent(0) == 2);
  REQUIRE(M.extent(1) == 2);
  REQUIRE(Md.extent(0) == 2);
  REQUIRE(Md.extent(1) == 2);

  CHECK(M(0, 0) == 0);
  CHECK(M(0, 1) == 1);
  CHECK(M(1, 0) == 2);
  CHECK(M(1, 1) == 3);

  CHECK((M == Md));

  {
    zipper::Matrix m = M;
    auto a = m.as_span();
    auto b = m.as_const_span();
    CHECK((a == b));

    zipper::Matrix<int, 2, 2>::const_span_type d = a;
    CHECK((a == d));
  }

  // this last case WOULD be very cool, but seems to not work due to a parse
  // limitation in type deductions? In particular, gcc at least seems to
  // really want y to be the name of a variable of type VectorBase
  // VectorBase(y) = {4, 5};
  // CHECK(v(0) == 2);
  // CHECK(v(1) == 3);
}
TEST_CASE("test_span_view", "[vector][storage][dense][span]") {
  using zipper::VectorBase;
  {
    zipper::Vector<double, 3> x = {1, 2, 3};

    VectorBase y = x.expression().as_span();
    CHECK(x == y);
    VectorBase z = x.expression().as_std_span();
    CHECK(x == z);
  }
  {
    zipper::Vector<double, std::dynamic_extent> x = {1, 2, 3};

    static_assert(std::decay_t<decltype(x)>::extents_type::rank() == 1);

    VectorBase y = x.expression().as_span();
    CHECK(x == y);
    VectorBase z = x.expression().as_std_span();
    CHECK(x == z);
  }
}
