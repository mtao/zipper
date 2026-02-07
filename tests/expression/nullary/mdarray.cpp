#include "../../catch_include.hpp"
#include <zipper/expression/nullary/MDArray.hpp>
using namespace zipper::expression::nullary;
using namespace zipper;
TEST_CASE("test_mdarray", "[expression][nullary]") {
  auto a = MDArray<double, zipper::extents<zipper::dynamic_extent>>(
      create_dextents(3));
  auto b = MDArray<double, extents<2, 3>>{};
  auto c = MDArray<double, extents<>>();

  const auto &mpa = a.mapping();
  const auto &mpb = b.mapping();
  int x = mpa(3);
  int y = mpb(3, 5);

  static_assert(a.extents().rank() == 1);
  REQUIRE(a.extents().rank() == 1);
  CHECK(a.extent(0) == 3);
  REQUIRE(b.extents().rank() == 2);
  CHECK(b.extent(0) == 2);
  CHECK(b.extent(1) == 3);

  REQUIRE(c.extents().rank() == 0);

  // testing is mostly to make sure that values can be evaluated, probability
  // of getting the same values twice is basically none

  for (index_type j = 0; j < 3; ++j) {
    double u = a(j);
    double v = a(j);
    CHECK(u != v);
  }

  for (index_type j = 0; j < 2; ++j) {
    for (index_type k = 0; k < 3; ++k) {
      double u = b(j, k);
      double v = b(j, k);
      CHECK(u != v);
    }
  }

  // for an "infinite" view we can pass in random values
  CHECK(c(2) != c(2));
  CHECK(c(1, 2) != c(1, 2));
  CHECK(c(2, 3, 1, 2) != c(2, 3, 1, 2));
  CHECK(c() != c());
}
