
#include <complex>
#include <zipper/expression/nullary/StaticConstant.hpp>
#include <zipper/static_scalar.hpp>
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
    StaticConstant<double, 5.0, 3> c;
    REQUIRE(c.extents().rank() == 1);
    CHECK(c.extent(0) == 3);
    CHECK(c(0) == 5.0);
    CHECK(c(1) == 5.0);
    CHECK(c(2) == 5.0);
  }

  SECTION("static 2D extents, value=3") {
    StaticConstant<double, 3.0, 2, 4> c;
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
    StaticConstant<float, 7.0f, std::dynamic_extent> c(5);
    REQUIRE(c.extents().rank() == 1);
    CHECK(c.extent(0) == 5);
    for (index_type i = 0; i < 5; ++i) {
      CHECK(c(i) == 7.0f);
    }
  }

  SECTION("negative value") {
    StaticConstant<double, -2.0, 3> c;
    CHECK(c(0) == -2.0);
    CHECK(c(1) == -2.0);
    CHECK(c(2) == -2.0);
  }

  SECTION("static_value is accessible") {
    STATIC_REQUIRE(StaticConstant<double, 42.0, 3>::static_value == 42);
    STATIC_REQUIRE(Ones<double, 3>::static_value == 1);
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
// sizeof tests (zero storage for StaticConstant with static extents)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("StaticConstant sizeof",
          "[expression][nullary][static_constant][sizeof]") {
  // StaticConstant with static extents stores no data — the value is in the
  // type and extents are empty base classes.
  STATIC_REQUIRE(sizeof(StaticConstant<double, 0.0, 3>) == 1);
  STATIC_REQUIRE(sizeof(StaticConstant<double, 1.0, 3>) == 1);
  STATIC_REQUIRE(sizeof(StaticConstant<float, 42.0f, 3, 3>) == 1);
  STATIC_REQUIRE(sizeof(Ones<double, 3, 3>) == 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// make_owned tests
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("StaticConstant make_owned",
          "[expression][nullary][static_constant]") {
  StaticConstant<double, 5.0, 3> c;
  auto owned = c.make_owned();
  CHECK(owned(0) == 5.0);
  CHECK(owned(1) == 5.0);
  CHECK(owned(2) == 5.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Typed values; compile-time scalars are std::integral_constant
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("StaticConstant values are typed",
          "[expression][nullary][static_constant][static_scalar]") {
  // The value is written in the element type: one value, one type.
  STATIC_CHECK(std::is_same_v<StaticConstant<double, 1.0, 2, 2>, Ones<double, 2, 2>>);
  STATIC_CHECK(std::is_same_v<decltype(Ones<float, 3>::static_value), const float>);
  // Structural zeros are Zero's job: a StaticConstant never has an index set,
  // even when its value is zero.
  STATIC_CHECK_FALSE(zipper::expression::detail::ExpressionTraits<
                     StaticConstant<double, 0.0, 3>>::has_index_set);

  // Non-integral values for arithmetic types.
  StaticConstant<double, 0.5, 3> h;
  CHECK(h(0) == 0.5);
  CHECK(h(2) == 0.5);
  STATIC_CHECK_FALSE(zipper::expression::detail::ExpressionTraits<
                     StaticConstant<double, 0.5, 3>>::has_index_set);

  StaticConstant<float, -0.25f, std::dynamic_extent> q(4);
  CHECK(q.extent(0) == 4);
  CHECK(q(3) == -0.25f);

  Vector<double, 3> v;
  v = StaticConstant<double, 2.5, 3>{};
  CHECK(v(0) == 2.5);
  CHECK(v(2) == 2.5);
}

TEST_CASE("compile-time scalars", "[static_scalar]") {
  // std::integral_constant, for any type usable as a template argument.
  using half = std::integral_constant<double, 0.5>;
  STATIC_CHECK(concepts::StaticScalar<half>);
  STATIC_CHECK(concepts::StaticScalarOf<half, 0.5>);
  STATIC_CHECK_FALSE(concepts::StaticScalar<double>);

  // cw<v>: the type comes from the literal.
  STATIC_CHECK(std::is_same_v<decltype(cw<1.0>), const std::integral_constant<double, 1.0>>);
  STATIC_CHECK(std::is_same_v<decltype(cw<1>), const std::integral_constant<int, 1>>);
  STATIC_CHECK(concepts::StaticScalarOf<decltype(cw<1.0>), 1>);

  // Coefficients: run-time or compile-time, convertible to the value type.
  STATIC_CHECK(concepts::Coefficient<double, double>);
  STATIC_CHECK(concepts::Coefficient<float, double>);
  STATIC_CHECK(concepts::Coefficient<decltype(cw<-1.0>), double>);
  STATIC_CHECK(concepts::Coefficient<decltype(cw<1>), std::complex<double>>);

  // scalar_product stays compile-time when both factors are.
  STATIC_CHECK(std::is_same_v<decltype(scalar_product<double>(cw<2.0>, cw<0.5>)),
                              std::integral_constant<double, 1.0>>);
  STATIC_CHECK(scalar_product<double>(cw<1.0>, 3.0) == 3.0);
  STATIC_CHECK(scalar_product<double>(2.0, 3.0) == 6.0);
}
