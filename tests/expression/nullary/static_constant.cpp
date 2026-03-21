
#include <zipper/expression/nullary/StaticConstant.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>

#include "../../catch_include.hpp"

using namespace zipper::expression::nullary;
using namespace zipper;

// ═══════════════════════════════════════════════════════════════════════════
// StaticConstant basic tests
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("StaticConstant basic", "[expression][nullary][static_constant]") {
  SECTION("static extents, value=5") {
    StaticConstant<double, 5, 3> c;
    REQUIRE(c.extents().rank() == 1);
    CHECK(c.extent(0) == 3);
    CHECK(c(0) == 5.0);
    CHECK(c(1) == 5.0);
    CHECK(c(2) == 5.0);
  }

  SECTION("static 2D extents, value=3") {
    StaticConstant<double, 3, 2, 4> c;
    REQUIRE(c.extents().rank() == 2);
    CHECK(c.extent(0) == 2);
    CHECK(c.extent(1) == 4);
    for (index_type i = 0; i < 2; ++i) {
      for (index_type j = 0; j < 4; ++j) {
        CHECK(c(i, j) == 3.0);
      }
    }
  }

  SECTION("dynamic extent, value=7") {
    StaticConstant<float, 7, std::dynamic_extent> c(5);
    REQUIRE(c.extents().rank() == 1);
    CHECK(c.extent(0) == 5);
    for (index_type i = 0; i < 5; ++i) {
      CHECK(c(i) == 7.0f);
    }
  }

  SECTION("negative value") {
    StaticConstant<double, -2, 3> c;
    CHECK(c(0) == -2.0);
    CHECK(c(1) == -2.0);
    CHECK(c(2) == -2.0);
  }

  SECTION("static_value is accessible") {
    STATIC_REQUIRE(StaticConstant<double, 42, 3>::static_value == 42);
    STATIC_REQUIRE(Zero<double, 3>::static_value == 0);
    STATIC_REQUIRE(Ones<double, 3>::static_value == 1);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Zero tests
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Zero expression", "[expression][nullary][zero]") {
  SECTION("static 1D zero") {
    Zero<double, 4> z;
    REQUIRE(z.extents().rank() == 1);
    CHECK(z.extent(0) == 4);
    for (index_type i = 0; i < 4; ++i) {
      CHECK(z(i) == 0.0);
    }
  }

  SECTION("static 2D zero") {
    Zero<double, 3, 3> z;
    REQUIRE(z.extents().rank() == 2);
    for (index_type i = 0; i < 3; ++i) {
      for (index_type j = 0; j < 3; ++j) {
        CHECK(z(i, j) == 0.0);
      }
    }
  }

  SECTION("dynamic 1D zero") {
    Zero<float, std::dynamic_extent> z(6);
    CHECK(z.extent(0) == 6);
    for (index_type i = 0; i < 6; ++i) {
      CHECK(z(i) == 0.0f);
    }
  }

  SECTION("assign to vector") {
    Vector<double, 3> v;
    v(0) = 1.0;
    v(1) = 2.0;
    v(2) = 3.0;
    v = Zero<double, 3>{};
    CHECK(v(0) == 0.0);
    CHECK(v(1) == 0.0);
    CHECK(v(2) == 0.0);
  }

  SECTION("assign to matrix") {
    Matrix<double, 2, 2> m;
    m(0, 0) = 1.0;
    m(0, 1) = 2.0;
    m(1, 0) = 3.0;
    m(1, 1) = 4.0;
    m = Zero<double, 2, 2>{};
    for (index_type i = 0; i < 2; ++i) {
      for (index_type j = 0; j < 2; ++j) {
        CHECK(m(i, j) == 0.0);
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Ones tests
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Ones expression", "[expression][nullary][ones]") {
  SECTION("static 1D ones") {
    Ones<double, 3> o;
    REQUIRE(o.extents().rank() == 1);
    CHECK(o.extent(0) == 3);
    for (index_type i = 0; i < 3; ++i) {
      CHECK(o(i) == 1.0);
    }
  }

  SECTION("static 2D ones") {
    Ones<double, 2, 3> o;
    REQUIRE(o.extents().rank() == 2);
    for (index_type i = 0; i < 2; ++i) {
      for (index_type j = 0; j < 3; ++j) {
        CHECK(o(i, j) == 1.0);
      }
    }
  }

  SECTION("assign to vector") {
    Vector<double, 4> v;
    v = Ones<double, 4>{};
    for (index_type i = 0; i < 4; ++i) {
      CHECK(v(i) == 1.0);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Zero-aware sparsity (index_set) tests
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Zero index_set", "[expression][nullary][zero][sparsity]") {
  SECTION("1D zero has empty index_set") {
    Zero<double, 5> z;
    auto is = z.index_set<0>();
    CHECK(is.empty());
    CHECK(is.size() == 0);
    CHECK(!is.contains(0));
    CHECK(!is.contains(2));
  }

  SECTION("2D zero has empty index_set for both dims") {
    Zero<double, 3, 3> z;
    auto is0 = z.index_set<0>(1);
    auto is1 = z.index_set<1>(1);
    CHECK(is0.empty());
    CHECK(is1.empty());
    CHECK(is0.size() == 0);
    CHECK(is1.size() == 0);
  }

  SECTION("2D zero col_range_for_row / row_range_for_col") {
    Zero<double, 4, 4> z;
    auto cr = z.col_range_for_row(2);
    auto rr = z.row_range_for_col(1);
    CHECK(cr.empty());
    CHECK(rr.empty());
  }

  SECTION("non-zero StaticConstant has no index_set") {
    // Ones and other non-zero StaticConstants should NOT have index_set
    // (the has_index_set trait is false for Value != 0).
    using traits =
        zipper::expression::detail::ExpressionTraits<Ones<double, 3, 3>>;
    STATIC_REQUIRE(!traits::has_index_set);

    using zero_traits =
        zipper::expression::detail::ExpressionTraits<Zero<double, 3, 3>>;
    STATIC_REQUIRE(zero_traits::has_index_set);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// sizeof tests (zero storage for StaticConstant with static extents)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("StaticConstant sizeof",
          "[expression][nullary][static_constant][sizeof]") {
  // StaticConstant with static extents stores no data — the value is in the
  // type and extents are empty base classes.
  STATIC_REQUIRE(sizeof(StaticConstant<double, 0, 3>) == 1);
  STATIC_REQUIRE(sizeof(StaticConstant<double, 1, 3>) == 1);
  STATIC_REQUIRE(sizeof(StaticConstant<float, 42, 3, 3>) == 1);
  STATIC_REQUIRE(sizeof(Zero<double, 3>) == 1);
  STATIC_REQUIRE(sizeof(Ones<double, 3, 3>) == 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// make_owned tests
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("StaticConstant make_owned",
          "[expression][nullary][static_constant]") {
  StaticConstant<double, 5, 3> c;
  auto owned = c.make_owned();
  CHECK(owned(0) == 5.0);
  CHECK(owned(1) == 5.0);
  CHECK(owned(2) == 5.0);
}
