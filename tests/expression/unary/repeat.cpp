
#include <zipper/expression/nullary/MDArray.hpp>
#include <zipper/expression/unary/Repeat.hpp>

#include "../../catch_include.hpp"

using namespace zipper;

TEST_CASE("test_repeat_left_rank1", "[expression][unary][repeat]") {
  expression::nullary::MDArray<double, extents<3>,
                               storage::layout_left,
                               default_accessor_policy<double>>
      a;
  a(0) = 1.0;
  a(1) = 2.0;
  a(2) = 3.0;

  using expr_type = std::decay_t<decltype(a)>;
  // Left repeat with Count=1: adds 1 dynamic dimension on the left
  // Result has rank 2: (dynamic, 3)
  expression::unary::Repeat<expression::unary::RepeatMode::Left, 1, const expr_type> l(a);

  REQUIRE(l.extents().rank() == 2);
  // The repeated dimension has dynamic_extent (0 from the repeat helper)
  // The original dimension is static 3
  REQUIRE(l.static_extent(1) == 3);

  // l(k, j) should equal a(j) for any k
  for (index_type j = 0; j < a.extent(0); ++j) {
    for (index_type k = 0; k < 5; ++k) {
      CHECK(l(k, j) == a(j));
    }
  }
}

TEST_CASE("test_repeat_right_rank1", "[expression][unary][repeat]") {
  expression::nullary::MDArray<double, extents<3>,
                               storage::layout_left,
                               default_accessor_policy<double>>
      a;
  a(0) = 10.0;
  a(1) = 20.0;
  a(2) = 30.0;

  using expr_type = std::decay_t<decltype(a)>;
  // Right repeat with Count=1: adds 1 dynamic dimension on the right
  // Result has rank 2: (3, dynamic)
  expression::unary::Repeat<expression::unary::RepeatMode::Right, 1, const expr_type> r(a);

  REQUIRE(r.extents().rank() == 2);
  REQUIRE(r.static_extent(0) == 3);

  // r(j, k) should equal a(j) for any k
  for (index_type j = 0; j < a.extent(0); ++j) {
    for (index_type k = 0; k < 5; ++k) {
      CHECK(r(j, k) == a(j));
    }
  }
}

TEST_CASE("test_repeat_left_rank2", "[expression][unary][repeat]") {
  expression::nullary::MDArray<double, extents<2, 3>,
                               storage::layout_left,
                               default_accessor_policy<double>>
      a;
  a(0, 0) = 1.0;
  a(0, 1) = 2.0;
  a(0, 2) = 3.0;
  a(1, 0) = 4.0;
  a(1, 1) = 5.0;
  a(1, 2) = 6.0;

  using expr_type = std::decay_t<decltype(a)>;
  // Left repeat with Count=2: adds 2 dynamic dimensions on the left
  // Result has rank 4: (dynamic, dynamic, 2, 3)
  expression::unary::Repeat<expression::unary::RepeatMode::Left, 2, const expr_type> l(a);

  REQUIRE(l.extents().rank() == 4);
  REQUIRE(l.static_extent(2) == 2);
  REQUIRE(l.static_extent(3) == 3);

  for (index_type j = 0; j < a.extent(0); ++j) {
    for (index_type k = 0; k < a.extent(1); ++k) {
      for (index_type jj = 0; jj < 3; ++jj) {
        for (index_type kk = 0; kk < 3; ++kk) {
          CHECK(l(jj, kk, j, k) == a(j, k));
        }
      }
    }
  }
}

TEST_CASE("test_repeat_right_rank2", "[expression][unary][repeat]") {
  expression::nullary::MDArray<double, extents<2, 3>,
                               storage::layout_left,
                               default_accessor_policy<double>>
      a;
  a(0, 0) = 1.0;
  a(0, 1) = 2.0;
  a(0, 2) = 3.0;
  a(1, 0) = 4.0;
  a(1, 1) = 5.0;
  a(1, 2) = 6.0;

  using expr_type = std::decay_t<decltype(a)>;
  // Right repeat with Count=2: adds 2 dynamic dimensions on the right
  // Result has rank 4: (2, 3, dynamic, dynamic)
  expression::unary::Repeat<expression::unary::RepeatMode::Right, 2, const expr_type> r(a);

  REQUIRE(r.extents().rank() == 4);
  REQUIRE(r.static_extent(0) == 2);
  REQUIRE(r.static_extent(1) == 3);

  for (index_type j = 0; j < a.extent(0); ++j) {
    for (index_type k = 0; k < a.extent(1); ++k) {
      for (index_type jj = 0; jj < 3; ++jj) {
        for (index_type kk = 0; kk < 3; ++kk) {
          CHECK(r(j, k, jj, kk) == a(j, k));
        }
      }
    }
  }
}
