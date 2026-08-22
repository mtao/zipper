
#include <zipper/expression/nullary/Random.hpp>

#include "../../catch_include.hpp"
using namespace zipper::expression::nullary;
using namespace zipper;
TEST_CASE("test_random", "[expression][nullary]") {
  const std::default_random_engine generator{12345};
  auto a = uniform_random<double>(create_dextents(3), -2.0, 3.0, generator);
  auto a_repeat =
      uniform_random<double>(create_dextents(3), -2.0, 3.0, generator);
  auto b = uniform_random<double>(extents<2, 3>{}, -2.0, 3.0, generator);
  auto b_repeat =
      uniform_random<double>(extents<2, 3>{}, -2.0, 3.0, generator);
  auto c = uniform_random<double>({}, -2.0, 3.0, generator);
  auto c_repeat = uniform_random<double>({}, -2.0, 3.0, generator);

  REQUIRE(a.extents().rank() == 1);
  CHECK(a.extent(0) == 3);
  REQUIRE(b.extents().rank() == 2);
  CHECK(b.extent(0) == 2);
  CHECK(b.extent(1) == 3);

  REQUIRE(c.extents().rank() == 0);

  for (index_type j = 0; j < 3; ++j) {
    const double value = a(j);
    CHECK(value == a_repeat(j));
    CHECK(value >= -2.0);
    CHECK(value < 3.0);
  }

  for (index_type j = 0; j < 2; ++j) {
    for (index_type k = 0; k < 3; ++k) {
      const double value = b(j, k);
      CHECK(value == b_repeat(j, k));
      CHECK(value >= -2.0);
      CHECK(value < 3.0);
    }
  }

  CHECK(c(2) == c_repeat(2));
  CHECK(c(1, 2) == c_repeat(1, 2));
  CHECK(c(2, 3, 1, 2) == c_repeat(2, 3, 1, 2));
  CHECK(c() == c_repeat());
}
