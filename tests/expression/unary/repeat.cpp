
#include <zipper/expression/nullary/MDArray.hpp>
#include <zipper/expression/nullary/Unit.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/unary/Repeat.hpp>
#include <zipper/expression/detail/IndexSet.hpp>

#include "../../catch_include.hpp"

using namespace zipper;
using namespace zipper::expression;
using namespace zipper::expression::detail;

TEST_CASE("test_repeat_left_rank1", "[expression][unary][repeat]") {
  expression::nullary::MDArray<double, extents<3>,
                               storage::layout_left,
                               default_accessor_policy<double>>
      a;
  a(0) = 1.0;
  a(1) = 2.0;
  a(2) = 3.0;

  using expr_type = std::decay_t<decltype(a)>;
  // Left repeat with Count=1: adds 1 dynamic dimension on the left
  // Result has rank 2: (dynamic, 3)
  expression::unary::Repeat<expression::unary::RepeatMode::Left, 1, const expr_type> l(a);

  REQUIRE(l.extents().rank() == 2);
  // The repeated dimension has dynamic_extent (0 from the repeat helper)
  // The original dimension is static 3
  REQUIRE(l.static_extent(1) == 3);

  // l(k, j) should equal a(j) for any k
  for (index_type j = 0; j < a.extent(0); ++j) {
    for (index_type k = 0; k < 5; ++k) {
      CHECK(l(k, j) == a(j));
    }
  }
}

TEST_CASE("test_repeat_right_rank1", "[expression][unary][repeat]") {
  expression::nullary::MDArray<double, extents<3>,
                               storage::layout_left,
                               default_accessor_policy<double>>
      a;
  a(0) = 10.0;
  a(1) = 20.0;
  a(2) = 30.0;

  using expr_type = std::decay_t<decltype(a)>;
  // Right repeat with Count=1: adds 1 dynamic dimension on the right
  // Result has rank 2: (3, dynamic)
  expression::unary::Repeat<expression::unary::RepeatMode::Right, 1, const expr_type> r(a);

  REQUIRE(r.extents().rank() == 2);
  REQUIRE(r.static_extent(0) == 3);

  // r(j, k) should equal a(j) for any k
  for (index_type j = 0; j < a.extent(0); ++j) {
    for (index_type k = 0; k < 5; ++k) {
      CHECK(r(j, k) == a(j));
    }
  }
}

TEST_CASE("test_repeat_left_rank2", "[expression][unary][repeat]") {
  expression::nullary::MDArray<double, extents<2, 3>,
                               storage::layout_left,
                               default_accessor_policy<double>>
      a;
  a(0, 0) = 1.0;
  a(0, 1) = 2.0;
  a(0, 2) = 3.0;
  a(1, 0) = 4.0;
  a(1, 1) = 5.0;
  a(1, 2) = 6.0;

  using expr_type = std::decay_t<decltype(a)>;
  // Left repeat with Count=2: adds 2 dynamic dimensions on the left
  // Result has rank 4: (dynamic, dynamic, 2, 3)
  expression::unary::Repeat<expression::unary::RepeatMode::Left, 2, const expr_type> l(a);

  REQUIRE(l.extents().rank() == 4);
  REQUIRE(l.static_extent(2) == 2);
  REQUIRE(l.static_extent(3) == 3);

  for (index_type j = 0; j < a.extent(0); ++j) {
    for (index_type k = 0; k < a.extent(1); ++k) {
      for (index_type jj = 0; jj < 3; ++jj) {
        for (index_type kk = 0; kk < 3; ++kk) {
          CHECK(l(jj, kk, j, k) == a(j, k));
        }
      }
    }
  }
}

TEST_CASE("test_repeat_right_rank2", "[expression][unary][repeat]") {
  expression::nullary::MDArray<double, extents<2, 3>,
                               storage::layout_left,
                               default_accessor_policy<double>>
      a;
  a(0, 0) = 1.0;
  a(0, 1) = 2.0;
  a(0, 2) = 3.0;
  a(1, 0) = 4.0;
  a(1, 1) = 5.0;
  a(1, 2) = 6.0;

  using expr_type = std::decay_t<decltype(a)>;
  // Right repeat with Count=2: adds 2 dynamic dimensions on the right
  // Result has rank 4: (2, 3, dynamic, dynamic)
  expression::unary::Repeat<expression::unary::RepeatMode::Right, 2, const expr_type> r(a);

  REQUIRE(r.extents().rank() == 4);
  REQUIRE(r.static_extent(0) == 2);
  REQUIRE(r.static_extent(1) == 3);

  for (index_type j = 0; j < a.extent(0); ++j) {
    for (index_type k = 0; k < a.extent(1); ++k) {
      for (index_type jj = 0; jj < 3; ++jj) {
        for (index_type kk = 0; kk < 3; ++kk) {
          CHECK(r(j, k, jj, kk) == a(j, k));
        }
      }
    }
  }
}

// ════════════════════════════════════════════════════════════════════════════
// IndexSet propagation tests
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("repeat_has_index_set_propagation", "[expression][unary][repeat][indexset]") {
    // MDArray does NOT have index_set
    using dense_type = expression::nullary::MDArray<double, extents<3>,
        storage::layout_left, default_accessor_policy<double>>;
    using dense_repeat = unary::Repeat<unary::RepeatMode::Left, 1, const dense_type>;
    STATIC_REQUIRE_FALSE(ExpressionTraits<dense_repeat>::has_index_set);

    // Unit vector DOES have index_set
    using unit_type = std::decay_t<decltype(expression::nullary::unit_vector<double, 5, 2>())>;
    using unit_repeat = unary::Repeat<unary::RepeatMode::Left, 1, const unit_type>;
    STATIC_REQUIRE(ExpressionTraits<unit_repeat>::has_index_set);
    STATIC_REQUIRE(ExpressionTraits<unit_repeat>::has_known_zeros);

    // Identity matrix DOES have index_set
    using identity_type = expression::nullary::Identity<double, 4, 4>;
    using identity_repeat_left = unary::Repeat<unary::RepeatMode::Left, 1, const identity_type>;
    STATIC_REQUIRE(ExpressionTraits<identity_repeat_left>::has_index_set);

    using identity_repeat_right = unary::Repeat<unary::RepeatMode::Right, 2, const identity_type>;
    STATIC_REQUIRE(ExpressionTraits<identity_repeat_right>::has_index_set);
}

TEST_CASE("repeat_left_unit_index_set", "[expression][unary][repeat][indexset]") {
    // Unit vector e_2 in R^5: nonzero only at index 2
    auto unit = expression::nullary::unit_vector<double, 5, 2>();
    using unit_type = std::decay_t<decltype(unit)>;

    // Repeat Left: rank-1 → rank-2, dims = (dynamic, 5)
    // Output dim 0 = repeated (broadcast), dim 1 = child dim 0
    unary::Repeat<unary::RepeatMode::Left, 1, const unit_type> rep(unit);

    REQUIRE(rep.extents().rank() == 2);

    SECTION("child dimension propagates Unit sparsity") {
        // index_set<1>(any_row) should give the Unit's nonzero index = {2}
        auto is1 = rep.index_set<1>(0);
        CHECK(is1.size() == 1);
        CHECK(is1.contains(2));
        CHECK_FALSE(is1.contains(0));
        CHECK_FALSE(is1.contains(1));
        CHECK_FALSE(is1.contains(3));
        CHECK_FALSE(is1.contains(4));

        // Same result for any row (broadcast)
        auto is1b = rep.index_set<1>(42);
        CHECK(is1b.size() == 1);
        CHECK(is1b.contains(2));
    }

    SECTION("repeated dimension returns FullRange") {
        auto is0 = rep.index_set<0>(2);
        // FullRange: contains() always returns true
        CHECK(is0.contains(0));
        CHECK(is0.contains(1));
        CHECK(is0.contains(999));
    }

    SECTION("convenience methods") {
        auto cols = rep.col_range_for_row(0);
        CHECK(cols.size() == 1);
        CHECK(cols.contains(2));

        auto rows = rep.row_range_for_col(2);
        CHECK(rows.contains(0));
        CHECK(rows.contains(100));
    }
}

TEST_CASE("repeat_right_unit_index_set", "[expression][unary][repeat][indexset]") {
    // Unit vector e_1 in R^4: nonzero only at index 1
    auto unit = expression::nullary::unit_vector<double, 4, 1>();
    using unit_type = std::decay_t<decltype(unit)>;

    // Repeat Right: rank-1 → rank-2, dims = (4, dynamic)
    // Output dim 0 = child dim 0, dim 1 = repeated (broadcast)
    unary::Repeat<unary::RepeatMode::Right, 1, const unit_type> rep(unit);

    REQUIRE(rep.extents().rank() == 2);

    SECTION("child dimension propagates Unit sparsity") {
        // index_set<0>(any_col) should give the Unit's nonzero index = {1}
        auto is0 = rep.index_set<0>(0);
        CHECK(is0.size() == 1);
        CHECK(is0.contains(1));
        CHECK_FALSE(is0.contains(0));
        CHECK_FALSE(is0.contains(2));
    }

    SECTION("repeated dimension returns FullRange") {
        auto is1 = rep.index_set<1>(1);
        CHECK(is1.contains(0));
        CHECK(is1.contains(42));
    }

    SECTION("convenience methods") {
        auto cols = rep.col_range_for_row(1);
        CHECK(cols.contains(0));
        CHECK(cols.contains(100));

        auto rows = rep.row_range_for_col(0);
        CHECK(rows.size() == 1);
        CHECK(rows.contains(1));
    }
}

TEST_CASE("repeat_left_identity_index_set", "[expression][unary][repeat][indexset]") {
    // Identity matrix 3x3: nonzero at (i,i)
    expression::nullary::Identity<double, 3, 3> eye;

    using eye_type = std::decay_t<decltype(eye)>;

    // Repeat Left with Count=1: rank-2 → rank-3
    // Output dims: (dynamic, 3, 3)
    // dim 0 = repeated, dim 1 = child dim 0, dim 2 = child dim 1
    unary::Repeat<unary::RepeatMode::Left, 1, const eye_type> rep(eye);

    REQUIRE(rep.extents().rank() == 3);

    SECTION("repeated dimension is fully dense") {
        auto is0 = rep.index_set<0>(0);
        CHECK(is0.contains(0));
        CHECK(is0.contains(999));
    }

    SECTION("child dimensions propagate Identity sparsity") {
        // index_set<1>(col=1) should return {1} (Identity: row == col)
        auto is1 = rep.index_set<1>(1);
        CHECK(is1.size() == 1);
        CHECK(is1.contains(1));
        CHECK_FALSE(is1.contains(0));
        CHECK_FALSE(is1.contains(2));

        // index_set<2>(row=0) should return {0}
        auto is2 = rep.index_set<2>(0);
        CHECK(is2.size() == 1);
        CHECK(is2.contains(0));
        CHECK_FALSE(is2.contains(1));
    }
}

TEST_CASE("repeat_right_identity_index_set", "[expression][unary][repeat][indexset]") {
    expression::nullary::Identity<double, 3, 3> eye;
    using eye_type = std::decay_t<decltype(eye)>;

    // Repeat Right with Count=2: rank-2 → rank-4
    // Output dims: (3, 3, dynamic, dynamic)
    // dim 0,1 = child dims, dim 2,3 = repeated
    unary::Repeat<unary::RepeatMode::Right, 2, const eye_type> rep(eye);

    REQUIRE(rep.extents().rank() == 4);

    SECTION("child dimensions propagate Identity sparsity") {
        // index_set<0>(col=2) should return {2}
        auto is0 = rep.index_set<0>(2);
        CHECK(is0.size() == 1);
        CHECK(is0.contains(2));

        // index_set<1>(row=0) should return {0}
        auto is1 = rep.index_set<1>(0);
        CHECK(is1.size() == 1);
        CHECK(is1.contains(0));
    }

    SECTION("repeated dimensions are fully dense") {
        auto is2 = rep.index_set<2>(0);
        CHECK(is2.contains(0));
        CHECK(is2.contains(42));

        auto is3 = rep.index_set<3>(0);
        CHECK(is3.contains(0));
        CHECK(is3.contains(42));
    }
}

TEST_CASE("repeat_count0_identity_index_set", "[expression][unary][repeat][indexset]") {
    // Count=0: identity-like, no dimensions added
    expression::nullary::Identity<double, 3, 3> eye;
    using eye_type = std::decay_t<decltype(eye)>;

    unary::Repeat<unary::RepeatMode::Left, 0, const eye_type> rep(eye);

    REQUIRE(rep.extents().rank() == 2);

    // All dimensions are child dimensions, sparsity fully preserved
    auto cols = rep.col_range_for_row(1);
    CHECK(cols.size() == 1);
    CHECK(cols.contains(1));
    CHECK_FALSE(cols.contains(0));

    auto rows = rep.row_range_for_col(2);
    CHECK(rows.size() == 1);
    CHECK(rows.contains(2));
}

TEST_CASE("repeat_count0_unit_rank1_index_set", "[expression][unary][repeat][indexset]") {
    // Count=0 on rank-1: identity, output is still rank-1
    auto unit = expression::nullary::unit_vector<double, 5, 3>();
    using unit_type = std::decay_t<decltype(unit)>;

    unary::Repeat<unary::RepeatMode::Left, 0, const unit_type> rep(unit);

    REQUIRE(rep.extents().rank() == 1);

    // rank-1 index_set<0>() with no args
    auto seg = rep.index_set<0>();
    CHECK(seg.size() == 1);
    CHECK(seg.contains(3));
    CHECK_FALSE(seg.contains(0));

    // nonzero_segment convenience
    auto seg2 = rep.nonzero_segment();
    CHECK(seg2.size() == 1);
    CHECK(seg2.contains(3));
}
