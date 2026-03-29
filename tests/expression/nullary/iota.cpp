
#include <zipper/expression/nullary/Iota.hpp>

#include "../../catch_include.hpp"

using namespace zipper::expression::nullary;
using namespace zipper;

TEST_CASE("test_iota_rank1_static", "[expression][nullary]") {
  Iota<double, 0, 5> v;

  REQUIRE(v.extents().rank() == 1);
  CHECK(v.extent(0) == 5);

  CHECK(v.coeff(index_type(0)) == 0.0);
  CHECK(v.coeff(index_type(1)) == 1.0);
  CHECK(v.coeff(index_type(2)) == 2.0);
  CHECK(v.coeff(index_type(3)) == 3.0);
  CHECK(v.coeff(index_type(4)) == 4.0);
}

TEST_CASE("test_iota_rank1_dynamic", "[expression][nullary]") {
  Iota<double, 0, dynamic_extent> v(5);

  REQUIRE(v.extents().rank() == 1);
  CHECK(v.extent(0) == 5);

  CHECK(v.coeff(index_type(0)) == 0.0);
  CHECK(v.coeff(index_type(1)) == 1.0);
  CHECK(v.coeff(index_type(4)) == 4.0);
}

TEST_CASE("test_iota_rank2_axis0", "[expression][nullary]") {
  Iota<double, 0, 3, 4> m;

  REQUIRE(m.extents().rank() == 2);
  CHECK(m.extent(0) == 3);
  CHECK(m.extent(1) == 4);

  // coeff(i, j) should return i (the row index)
  for (index_type i = 0; i < 3; ++i) {
    for (index_type j = 0; j < 4; ++j) {
      CHECK(m.coeff(i, j) == static_cast<double>(i));
    }
  }
}

TEST_CASE("test_iota_rank2_axis1", "[expression][nullary]") {
  Iota<double, 1, 3, 4> m;

  REQUIRE(m.extents().rank() == 2);

  // coeff(i, j) should return j (the column index)
  for (index_type i = 0; i < 3; ++i) {
    for (index_type j = 0; j < 4; ++j) {
      CHECK(m.coeff(i, j) == static_cast<double>(j));
    }
  }
}

TEST_CASE("test_iota_rank3", "[expression][nullary]") {
  Iota<int, 2, 2, 3, 4> t;

  REQUIRE(t.extents().rank() == 3);
  CHECK(t.extent(2) == 4);

  // coeff(i, j, k) should return k (axis 2)
  for (index_type i = 0; i < 2; ++i) {
    for (index_type j = 0; j < 3; ++j) {
      for (index_type k = 0; k < 4; ++k) {
        CHECK(t.coeff(i, j, k) == static_cast<int>(k));
      }
    }
  }
}

TEST_CASE("test_iota_int_type", "[expression][nullary]") {
  Iota<int, 0, 4> v;

  CHECK(v.coeff(index_type(0)) == 0);
  CHECK(v.coeff(index_type(3)) == 3);
}

TEST_CASE("test_iota_make_owned", "[expression][nullary]") {
  Iota<double, 0, 5> v;
  auto owned = v.make_owned();

  CHECK(owned.coeff(index_type(3)) == 3.0);
  CHECK(owned.extent(0) == 5);
}

// ── iota() free function tests ──────────────────────────────────────

TEST_CASE("test_iota_with_start_static", "[expression][nullary]") {
  auto v = iota<double>(3.0, extents<5>{});

  CHECK(v.coeff(index_type(0)) == Catch::Approx(3.0));
  CHECK(v.coeff(index_type(1)) == Catch::Approx(4.0));
  CHECK(v.coeff(index_type(2)) == Catch::Approx(5.0));
  CHECK(v.coeff(index_type(3)) == Catch::Approx(6.0));
  CHECK(v.coeff(index_type(4)) == Catch::Approx(7.0));
}

TEST_CASE("test_iota_with_start_dynamic", "[expression][nullary]") {
  auto v = iota<double>(10.0, create_dextents(4));

  CHECK(v.coeff(index_type(0)) == Catch::Approx(10.0));
  CHECK(v.coeff(index_type(1)) == Catch::Approx(11.0));
  CHECK(v.coeff(index_type(2)) == Catch::Approx(12.0));
  CHECK(v.coeff(index_type(3)) == Catch::Approx(13.0));
}

TEST_CASE("test_iota_with_start_zero", "[expression][nullary]") {
  auto v = iota<double>(0.0, extents<3>{});

  CHECK(v.coeff(index_type(0)) == Catch::Approx(0.0));
  CHECK(v.coeff(index_type(1)) == Catch::Approx(1.0));
  CHECK(v.coeff(index_type(2)) == Catch::Approx(2.0));
}

TEST_CASE("test_iota_with_start_negative", "[expression][nullary]") {
  auto v = iota<double>(-2.0, extents<5>{});

  CHECK(v.coeff(index_type(0)) == Catch::Approx(-2.0));
  CHECK(v.coeff(index_type(1)) == Catch::Approx(-1.0));
  CHECK(v.coeff(index_type(2)) == Catch::Approx(0.0));
  CHECK(v.coeff(index_type(3)) == Catch::Approx(1.0));
  CHECK(v.coeff(index_type(4)) == Catch::Approx(2.0));
}

TEST_CASE("test_iota_rank2_with_start", "[expression][nullary]") {
  // iota along axis 0 with start=5 on a 3x4 shape
  auto m = iota<double, 0>(5.0, extents<3, 4>{});

  // coeff(i, j) should return 5 + i
  for (index_type i = 0; i < 3; ++i) {
    for (index_type j = 0; j < 4; ++j) {
      CHECK(m.coeff(i, j) == Catch::Approx(5.0 + static_cast<double>(i)));
    }
  }
}

// ── linear_space() free function tests ──────────────────────────────

TEST_CASE("test_linear_space_basic", "[expression][nullary]") {
  // 5 evenly spaced values from 0 to 1: 0, 0.25, 0.5, 0.75, 1.0
  auto v = linear_space<double>(0.0, 1.0, extents<5>{});

  CHECK(v.coeff(index_type(0)) == Catch::Approx(0.0));
  CHECK(v.coeff(index_type(1)) == Catch::Approx(0.25));
  CHECK(v.coeff(index_type(2)) == Catch::Approx(0.5));
  CHECK(v.coeff(index_type(3)) == Catch::Approx(0.75));
  CHECK(v.coeff(index_type(4)) == Catch::Approx(1.0));
}

TEST_CASE("test_linear_space_negative_range", "[expression][nullary]") {
  // 3 values from -1 to 1: -1, 0, 1
  auto v = linear_space<double>(-1.0, 1.0, extents<3>{});

  CHECK(v.coeff(index_type(0)) == Catch::Approx(-1.0));
  CHECK(v.coeff(index_type(1)) == Catch::Approx(0.0));
  CHECK(v.coeff(index_type(2)) == Catch::Approx(1.0));
}

TEST_CASE("test_linear_space_descending", "[expression][nullary]") {
  // 4 values from 10 to 1: 10, 7, 4, 1
  auto v = linear_space<double>(10.0, 1.0, extents<4>{});

  CHECK(v.coeff(index_type(0)) == Catch::Approx(10.0));
  CHECK(v.coeff(index_type(1)) == Catch::Approx(7.0));
  CHECK(v.coeff(index_type(2)) == Catch::Approx(4.0));
  CHECK(v.coeff(index_type(3)) == Catch::Approx(1.0));
}

TEST_CASE("test_linear_space_two_points", "[expression][nullary]") {
  // 2 values from 0 to 10: just endpoints
  auto v = linear_space<double>(0.0, 10.0, extents<2>{});

  CHECK(v.coeff(index_type(0)) == Catch::Approx(0.0));
  CHECK(v.coeff(index_type(1)) == Catch::Approx(10.0));
}

TEST_CASE("test_linear_space_dynamic", "[expression][nullary]") {
  auto v = linear_space<double>(0.0, 4.0, create_dextents(5));

  CHECK(v.coeff(index_type(0)) == Catch::Approx(0.0));
  CHECK(v.coeff(index_type(1)) == Catch::Approx(1.0));
  CHECK(v.coeff(index_type(2)) == Catch::Approx(2.0));
  CHECK(v.coeff(index_type(3)) == Catch::Approx(3.0));
  CHECK(v.coeff(index_type(4)) == Catch::Approx(4.0));
}

TEST_CASE("test_linear_space_rank2_axis1", "[expression][nullary]") {
  // 3x5 matrix, linearly spaced along axis 1 (columns) from 0 to 1
  // So each row has values 0, 0.25, 0.5, 0.75, 1.0
  auto m = linear_space<double, 1>(0.0, 1.0, extents<3, 5>{});

  for (index_type i = 0; i < 3; ++i) {
    CHECK(m.coeff(i, index_type(0)) == Catch::Approx(0.0));
    CHECK(m.coeff(i, index_type(1)) == Catch::Approx(0.25));
    CHECK(m.coeff(i, index_type(2)) == Catch::Approx(0.5));
    CHECK(m.coeff(i, index_type(3)) == Catch::Approx(0.75));
    CHECK(m.coeff(i, index_type(4)) == Catch::Approx(1.0));
  }
}

TEST_CASE("test_linear_space_rank2_axis0", "[expression][nullary]") {
  // 4x3 matrix, linearly spaced along axis 0 (rows) from 0 to 3
  // So each column has values 0, 1, 2, 3
  auto m = linear_space<double, 0>(0.0, 3.0, extents<4, 3>{});

  for (index_type j = 0; j < 3; ++j) {
    CHECK(m.coeff(index_type(0), j) == Catch::Approx(0.0));
    CHECK(m.coeff(index_type(1), j) == Catch::Approx(1.0));
    CHECK(m.coeff(index_type(2), j) == Catch::Approx(2.0));
    CHECK(m.coeff(index_type(3), j) == Catch::Approx(3.0));
  }
}
