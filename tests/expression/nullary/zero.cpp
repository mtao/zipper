#include <complex>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/StaticConstant.hpp>
#include <zipper/expression/nullary/Zero.hpp>

#include "../../catch_include.hpp"

using namespace zipper::expression::nullary;
using namespace zipper;

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
// Any element type, storage, make_owned
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Zero of any element type", "[expression][nullary][zero]") {
  // std::complex cannot be a template argument; Zero does not need one.
  using C = std::complex<double>;
  Zero<C, 2, 2> zc;
  CHECK(zc(1, 0) == C(0.0, 0.0));
  STATIC_CHECK(zipper::expression::detail::ExpressionTraits<Zero<C, 2, 2>>::has_index_set);

  VectorX<C> v(3);
  for (index_type i = 0; i < 3; ++i) v(i) = C(1.0 + i, -1.0);
  v = Zero<C, dynamic_extent>(3);
  for (index_type i = 0; i < 3; ++i) CHECK(v(i) == C(0.0, 0.0));
}

TEST_CASE("Zero sizeof and make_owned", "[expression][nullary][zero]") {
  STATIC_REQUIRE(sizeof(Zero<double, 3>) == 1);
  STATIC_REQUIRE(sizeof(Zero<double, 3, 3>) == 1);
  Zero<double, dynamic_extent> z(4);
  auto owned = z.make_owned();
  CHECK(owned.extent(0) == 4);
  CHECK(owned(3) == 0.0);
}
