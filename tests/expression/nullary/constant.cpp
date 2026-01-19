
#include <zipper/expression/nullary/Constant.hpp>

#include "../../catch_include.hpp"

using namespace zipper::expression::nullary;
using namespace zipper;

TEST_CASE("test_constant", "[views][nullary]") {

  Constant a(double(0), create_dextents(3));
  Constant b(double(1), extents<2, 3>{});
  Constant c(double(2));

  REQUIRE(a.extents().rank() == 1);
  CHECK(a.extent(0) == 3);
  REQUIRE(b.extents().rank() == 2);
  CHECK(b.extent(0) == 2);
  CHECK(b.extent(1) == 3);

  REQUIRE(c.extents().rank() == 0);

  CHECK(a(0) == 0);
  CHECK(a(1) == 0);
  CHECK(a(2) == 0);

  CHECK(b(0, 0) == 1);
  CHECK(b(0, 1) == 1);
  CHECK(b(0, 2) == 1);
  CHECK(b(1, 0) == 1);
  CHECK(b(1, 1) == 1);
  CHECK(b(1, 2) == 1);

  // for an "infinite" view we can pass in random values
  CHECK(c(2) == 2);
  CHECK(c(1, 2) == 2);
  CHECK(c(2, 3, 1, 2) == 2);
  CHECK(c() == 2);
}
