#include "../../catch_include.hpp"
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/unary/ExtentView.hpp>

using namespace zipper;

// ════════════════════════════════════════════════════════════════════════
// as_dynamic: static → dynamic extent erasure
// ════════════════════════════════════════════════════════════════════════

TEST_CASE("as_dynamic_vector", "[expression][unary][extent_view]") {
  Vector<double, 3> v{1.0, 2.0, 3.0};

  auto dyn = as_dynamic(v.expression());

  // Extents should now be fully dynamic
  using dyn_extents = typename std::decay_t<decltype(dyn)>::extents_type;
  STATIC_CHECK(dyn_extents::rank() == 1);
  STATIC_CHECK(dyn_extents::rank_dynamic() == 1);

  // Runtime sizes preserved
  CHECK(dyn.extent(0) == 3);

  // Values preserved
  CHECK(dyn.coeff(0) == 1.0);
  CHECK(dyn.coeff(1) == 2.0);
  CHECK(dyn.coeff(2) == 3.0);
}

TEST_CASE("as_dynamic_matrix", "[expression][unary][extent_view]") {
  Matrix<double, 2, 3> m{{1, 2, 3}, {4, 5, 6}};

  auto dyn = as_dynamic(m.expression());

  using dyn_extents = typename std::decay_t<decltype(dyn)>::extents_type;
  STATIC_CHECK(dyn_extents::rank() == 2);
  STATIC_CHECK(dyn_extents::rank_dynamic() == 2);

  CHECK(dyn.extent(0) == 2);
  CHECK(dyn.extent(1) == 3);

  CHECK(dyn.coeff(0, 0) == 1);
  CHECK(dyn.coeff(0, 2) == 3);
  CHECK(dyn.coeff(1, 0) == 4);
  CHECK(dyn.coeff(1, 2) == 6);
}

TEST_CASE("as_dynamic_already_dynamic", "[expression][unary][extent_view]") {
  VectorX<double> v = {10.0, 20.0};

  auto dyn = as_dynamic(v.expression());

  using dyn_extents = typename std::decay_t<decltype(dyn)>::extents_type;
  STATIC_CHECK(dyn_extents::rank() == 1);
  STATIC_CHECK(dyn_extents::rank_dynamic() == 1);

  CHECK(dyn.extent(0) == 2);
  CHECK(dyn.coeff(0) == 10.0);
  CHECK(dyn.coeff(1) == 20.0);
}

// ════════════════════════════════════════════════════════════════════════
// as_extents: dynamic → static (or static → different static)
// ════════════════════════════════════════════════════════════════════════

TEST_CASE("as_extents_dynamic_to_static_vector",
          "[expression][unary][extent_view]") {
  VectorX<double> v = {1.0, 2.0, 3.0};

  auto stat = as_extents<extents<3>>(v.expression());

  using stat_extents = typename std::decay_t<decltype(stat)>::extents_type;
  STATIC_CHECK(stat_extents::rank() == 1);
  STATIC_CHECK(stat_extents::rank_dynamic() == 0);
  STATIC_CHECK(stat_extents::static_extent(0) == 3);

  CHECK(stat.extent(0) == 3);
  CHECK(stat.coeff(0) == 1.0);
  CHECK(stat.coeff(1) == 2.0);
  CHECK(stat.coeff(2) == 3.0);
}

TEST_CASE("as_extents_dynamic_to_static_matrix",
          "[expression][unary][extent_view]") {
  MatrixXX<double> m(2, 3);
  m(0, 0) = 1;
  m(0, 1) = 2;
  m(0, 2) = 3;
  m(1, 0) = 4;
  m(1, 1) = 5;
  m(1, 2) = 6;

  auto stat = as_extents<extents<2, 3>>(m.expression());

  using stat_extents = typename std::decay_t<decltype(stat)>::extents_type;
  STATIC_CHECK(stat_extents::rank() == 2);
  STATIC_CHECK(stat_extents::rank_dynamic() == 0);
  STATIC_CHECK(stat_extents::static_extent(0) == 2);
  STATIC_CHECK(stat_extents::static_extent(1) == 3);

  CHECK(stat.coeff(0, 0) == 1);
  CHECK(stat.coeff(1, 2) == 6);
}

TEST_CASE("as_extents_mixed", "[expression][unary][extent_view]") {
  // Convert from fully dynamic to partially static
  MatrixXX<double> m(3, 4);
  m(0, 0) = 42;

  // extents<3, dynamic_extent> — first dim static, second dynamic
  auto mixed = as_extents<extents<3, dynamic_extent>>(m.expression());

  using mixed_extents = typename std::decay_t<decltype(mixed)>::extents_type;
  STATIC_CHECK(mixed_extents::rank() == 2);
  STATIC_CHECK(mixed_extents::rank_dynamic() == 1);
  STATIC_CHECK(mixed_extents::static_extent(0) == 3);
  STATIC_CHECK(mixed_extents::static_extent(1) == dynamic_extent);

  CHECK(mixed.extent(0) == 3);
  CHECK(mixed.extent(1) == 4);
  CHECK(mixed.coeff(0, 0) == 42);
}

TEST_CASE("as_extents_static_to_static_identity",
          "[expression][unary][extent_view]") {
  // Static → same static (should be a no-op semantically)
  Vector<double, 4> v{10, 20, 30, 40};

  auto same = as_extents<extents<4>>(v.expression());

  using same_extents = typename std::decay_t<decltype(same)>::extents_type;
  STATIC_CHECK(same_extents::rank() == 1);
  STATIC_CHECK(same_extents::rank_dynamic() == 0);
  STATIC_CHECK(same_extents::static_extent(0) == 4);

  CHECK(same.coeff(0) == 10);
  CHECK(same.coeff(3) == 40);
}

// ════════════════════════════════════════════════════════════════════════
// Round-trip: static → dynamic → static
// ════════════════════════════════════════════════════════════════════════

TEST_CASE("round_trip_static_dynamic_static",
          "[expression][unary][extent_view]") {
  Matrix<double, 3, 3> m{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}};

  // Erase to dynamic
  auto dyn = as_dynamic(m.expression());
  STATIC_CHECK(
      std::decay_t<decltype(dyn)>::extents_type::rank_dynamic() == 2);

  // Restore to static
  auto stat = as_extents<extents<3, 3>>(dyn);
  STATIC_CHECK(
      std::decay_t<decltype(stat)>::extents_type::rank_dynamic() == 0);

  CHECK(stat.coeff(0, 0) == 1);
  CHECK(stat.coeff(1, 1) == 2);
  CHECK(stat.coeff(2, 2) == 3);
  CHECK(stat.coeff(0, 1) == 0);
}

// ════════════════════════════════════════════════════════════════════════
// Writability: extent view on mutable expression preserves mutability
// ════════════════════════════════════════════════════════════════════════

TEST_CASE("extent_view_writability", "[expression][unary][extent_view]") {
  Vector<double, 3> v{0.0, 0.0, 0.0};

  auto dyn = as_dynamic(v.expression());

  // Write through the view
  dyn.coeff_ref(0) = 10.0;
  dyn.coeff_ref(1) = 20.0;
  dyn.coeff_ref(2) = 30.0;

  // Changes visible in original
  CHECK(v(0) == 10.0);
  CHECK(v(1) == 20.0);
  CHECK(v(2) == 30.0);
}

TEST_CASE("extent_view_const_correctness", "[expression][unary][extent_view]") {
  const Vector<double, 3> v{1.0, 2.0, 3.0};

  auto dyn = as_dynamic(v.expression());

  // Should still be readable
  CHECK(dyn.coeff(0) == 1.0);
  CHECK(dyn.coeff(1) == 2.0);
  CHECK(dyn.coeff(2) == 3.0);

  // Const expression: coeff_ref should not compile.
  // (We can't directly test a compile error, but we can verify
  //  the const qualification is propagated through the traits.)
  using dyn_traits =
      expression::detail::ExpressionTraits<std::decay_t<decltype(dyn)>>;
  STATIC_CHECK(dyn_traits::access_features.is_const);
}

// ════════════════════════════════════════════════════════════════════════
// make_owned: deep-copy semantics
// ════════════════════════════════════════════════════════════════════════

TEST_CASE("extent_view_make_owned", "[expression][unary][extent_view]") {
  auto make_view = []() {
    Vector<double, 3> v{100.0, 200.0, 300.0};
    auto dyn = as_dynamic(v.expression());
    return dyn.make_owned();
  };

  auto owned = make_view();
  // v is destroyed, but owned should still be valid
  CHECK(owned.coeff(0) == 100.0);
  CHECK(owned.coeff(1) == 200.0);
  CHECK(owned.coeff(2) == 300.0);
}
