#include "../../catch_include.hpp"
#include <zipper/Vector.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/expression/unary/Reshape.hpp>

using namespace zipper;

TEST_CASE("test_reshape_1d_to_2d", "[expression][unary][reshape]") {
  // Reshape a 6-element vector into a 2x3 matrix
  Vector<double, 6> v{1, 2, 3, 4, 5, 6};

  using new_extents = zipper::extents<2, 3>;
  auto r = expression::unary::Reshape(v.expression(), new_extents{});

  // layout_right (row-major):
  // v[0]=1 v[1]=2 v[2]=3 v[3]=4 v[4]=5 v[5]=6
  // becomes:
  // [[1, 2, 3],
  //  [4, 5, 6]]
  CHECK(r(0, 0) == 1);
  CHECK(r(0, 1) == 2);
  CHECK(r(0, 2) == 3);
  CHECK(r(1, 0) == 4);
  CHECK(r(1, 1) == 5);
  CHECK(r(1, 2) == 6);
}

TEST_CASE("test_reshape_2d_to_1d", "[expression][unary][reshape]") {
  // Reshape a 2x3 matrix into a 6-element vector
  Matrix<double, 2, 3> m{{1, 2, 3}, {4, 5, 6}};

  using new_extents = zipper::extents<6>;
  auto r = expression::unary::Reshape(m.expression(), new_extents{});

  CHECK(r(0) == 1);
  CHECK(r(1) == 2);
  CHECK(r(2) == 3);
  CHECK(r(3) == 4);
  CHECK(r(4) == 5);
  CHECK(r(5) == 6);
}

TEST_CASE("test_reshape_2d_to_2d", "[expression][unary][reshape]") {
  // Reshape a 2x3 matrix into a 3x2 matrix (not transpose!)
  Matrix<double, 2, 3> m{{1, 2, 3}, {4, 5, 6}};

  using new_extents = zipper::extents<3, 2>;
  auto r = expression::unary::Reshape(m.expression(), new_extents{});

  // Row-major reinterpretation:
  // [1,2,3,4,5,6] → [[1,2],[3,4],[5,6]]
  CHECK(r(0, 0) == 1);
  CHECK(r(0, 1) == 2);
  CHECK(r(1, 0) == 3);
  CHECK(r(1, 1) == 4);
  CHECK(r(2, 0) == 5);
  CHECK(r(2, 1) == 6);
}

TEST_CASE("test_reshape_static_construct", "[expression][unary][reshape]") {
  // Test the static-extents constructor (no extents argument needed)
  Vector<double, 4> v{10, 20, 30, 40};

  using new_extents = zipper::extents<2, 2>;
  auto r = expression::unary::Reshape<
      std::decay_t<decltype(v.expression())>, new_extents>(v.expression());

  CHECK(r(0, 0) == 10);
  CHECK(r(0, 1) == 20);
  CHECK(r(1, 0) == 30);
  CHECK(r(1, 1) == 40);
}

// ── Metadata-only reshape: flat/layout capability forwarding ────────────

#include <zipper/expression/concepts/capabilities.hpp>

TEST_CASE("test_reshape_linear_array_forwarding",
          "[expression][unary][reshape][layout]") {
  namespace c = zipper::expression::concepts;
  Matrix<double, 2, 6> m{{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}};

  using new_extents = zipper::extents<3, 4>;
  auto r = expression::unary::Reshape(m.expression(), new_extents{});
  using R = std::decay_t<decltype(r)>;

  // A reshape of a row-major dense leaf is pure metadata: it is itself a
  // LinearArray (buffer forwarded) and flat-vectorizable.
  STATIC_CHECK(c::LinearArray<R>);
  STATIC_CHECK(c::HasLayoutMapping<R>);
  STATIC_CHECK(c::FlatVectorizable<R>);

  // Same buffer, no data movement.
  CHECK(r.data() == m.expression().data());

  // The LinearArray contract over the NEW extents:
  // r[ r.mapping()(i,j) ] == r(i,j).
  const auto& map = r.mapping();
  for (index_type i = 0; i < r.extent(0); ++i)
    for (index_type j = 0; j < r.extent(1); ++j)
      CHECK(r[map(i, j)] == r(i, j));

  // Flat order is the child's flat order.
  for (index_type k = 0; k < 12; ++k) CHECK(r[k] == double(k + 1));
}

TEST_CASE("test_reshape_chain_collapses",
          "[expression][unary][reshape][layout]") {
  namespace c = zipper::expression::concepts;
  Matrix<double, 2, 6> m{{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}};

  // reshape(reshape(m)): both levels are metadata-only, so the chain is
  // still a LinearArray over the ROOT buffer — one composed index transform,
  // not a linearize→unravel round-trip per level.
  auto r1 = expression::unary::Reshape(m.expression(), zipper::extents<3, 4>{});
  auto r2 = expression::unary::Reshape(r1, zipper::extents<12>{});
  using R2 = std::decay_t<decltype(r2)>;
  STATIC_CHECK(c::LinearArray<R2>);
  CHECK(r2.data() == m.expression().data());
  for (index_type k = 0; k < 12; ++k) {
    CHECK(r2[k] == double(k + 1));
    CHECK(r2(k) == double(k + 1));
  }
}

TEST_CASE("test_reshape_column_major_child_falls_back",
          "[expression][unary][reshape][layout]") {
  namespace c = zipper::expression::concepts;
  // A column-major child's flat order does NOT match reshape's row-major
  // unravel semantics: no flat forwarding, but coeff access stays correct.
  Matrix<double, 2, 3, false> m{{1, 2, 3}, {4, 5, 6}};

  auto r = expression::unary::Reshape(m.expression(), zipper::extents<6>{});
  using R = std::decay_t<decltype(r)>;
  STATIC_CHECK(!c::FlatVectorizable<R>);
  STATIC_CHECK(!c::LinearArray<R>);
  STATIC_CHECK(!c::HasLayoutMapping<R>);

  // Row-major reshape semantics preserved regardless of child layout.
  Matrix<double, 2, 3> row{{1, 2, 3}, {4, 5, 6}};
  auto rr = expression::unary::Reshape(row.expression(), zipper::extents<6>{});
  for (index_type k = 0; k < 6; ++k) CHECK(r(k) == rr(k));
}
