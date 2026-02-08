#include "../../catch_include.hpp"
#include <zipper/expression/nullary/MDArray.hpp>
using namespace zipper::expression::nullary;
using namespace zipper;
TEST_CASE("test_mdarray", "[expression][nullary]") {
  // Dynamic 1D array
  auto a = MDArray<double, zipper::extents<zipper::dynamic_extent>>(
      create_dextents(3));

  // Static 2D array
  auto b = MDArray<double, extents<2, 3>>{};

  // Rank-0 (scalar) array
  auto c = MDArray<double, extents<>>();

  // Check extents
  static_assert(a.extents().rank() == 1);
  REQUIRE(a.extents().rank() == 1);
  CHECK(a.extent(0) == 3);

  REQUIRE(b.extents().rank() == 2);
  CHECK(b.extent(0) == 2);
  CHECK(b.extent(1) == 3);

  REQUIRE(c.extents().rank() == 0);

  // Test 1D: write values then read them back
  for (index_type j = 0; j < 3; ++j) {
    a.coeff_ref(j) = static_cast<double>(j + 1);
  }
  for (index_type j = 0; j < 3; ++j) {
    CHECK(a(j) == static_cast<double>(j + 1));
  }
  // Reading same index twice should give same value
  CHECK(a(0) == a(0));
  CHECK(a(1) == a(1));

  // Test 2D: write values then read them back
  for (index_type j = 0; j < 2; ++j) {
    for (index_type k = 0; k < 3; ++k) {
      b.coeff_ref(j, k) = static_cast<double>(j * 3 + k + 10);
    }
  }
  for (index_type j = 0; j < 2; ++j) {
    for (index_type k = 0; k < 3; ++k) {
      CHECK(b(j, k) == static_cast<double>(j * 3 + k + 10));
    }
  }

  // Test rank-0: write and read scalar
  c.coeff_ref() = 42.0;
  CHECK(c() == 42.0);
  // Rank-0 ignores indices — always accesses the single element
  CHECK(c(2) == 42.0);
  CHECK(c(1, 2) == 42.0);
  CHECK(c(2, 3, 1, 2) == 42.0);
}
