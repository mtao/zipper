
#include <zipper/expression/nullary/MDArray.hpp>
#include <zipper/expression/nullary/Unit.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/unary/Lift.hpp>
#include <zipper/expression/detail/IndexSet.hpp>

#include "../../catch_include.hpp"

using namespace zipper;
using namespace zipper::expression;
using namespace zipper::expression::detail;

TEST_CASE("lift_rank1", "[expression][unary][lift]") {
  expression::nullary::MDArray<double, extents<3>,
                               storage::layout_left,
                               default_accessor_policy<double>>
      a;
  a(0) = 10.0;
  a(1) = 20.0;
  a(2) = 30.0;

  using expr_type = std::decay_t<decltype(a)>;
  // Lift with Count=1: appends 1 dynamic dimension
  // Result has rank 2: (3, dynamic)
  expression::unary::Lift<1, const expr_type> l(a);

  REQUIRE(l.extents().rank() == 2);
  REQUIRE(l.static_extent(0) == 3);

  // l(j, k) should equal a(j) for any k
  for (index_type j = 0; j < a.extent(0); ++j) {
    for (index_type k = 0; k < 5; ++k) {
      CHECK(l(j, k) == a(j));
    }
  }
}

TEST_CASE("lift_rank2", "[expression][unary][lift]") {
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
  // Lift with Count=2: appends 2 dynamic dimensions
  // Result has rank 4: (2, 3, dynamic, dynamic)
  expression::unary::Lift<2, const expr_type> l(a);

  REQUIRE(l.extents().rank() == 4);
  REQUIRE(l.static_extent(0) == 2);
  REQUIRE(l.static_extent(1) == 3);

  for (index_type j = 0; j < a.extent(0); ++j) {
    for (index_type k = 0; k < a.extent(1); ++k) {
      for (index_type jj = 0; jj < 3; ++jj) {
        for (index_type kk = 0; kk < 3; ++kk) {
          CHECK(l(j, k, jj, kk) == a(j, k));
        }
      }
    }
  }
}

// ════════════════════════════════════════════════════════════════════════════
// IndexSet propagation tests
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("lift_has_index_set_propagation", "[expression][unary][lift][indexset]") {
    // MDArray does NOT have index_set
    using dense_type = expression::nullary::MDArray<double, extents<3>,
        storage::layout_left, default_accessor_policy<double>>;
    using dense_lift = unary::Lift<1, const dense_type>;
    STATIC_REQUIRE_FALSE(ExpressionTraits<dense_lift>::has_index_set);

    // Unit vector DOES have index_set
    using unit_type = std::decay_t<decltype(expression::nullary::unit_vector<double, 5, 2>())>;
    using unit_lift = unary::Lift<1, const unit_type>;
    STATIC_REQUIRE(ExpressionTraits<unit_lift>::has_index_set);
    STATIC_REQUIRE(ExpressionTraits<unit_lift>::has_known_zeros);

    // Identity matrix DOES have index_set
    using identity_type = expression::nullary::Identity<double, 4, 4>;
    using identity_lift = unary::Lift<2, const identity_type>;
    STATIC_REQUIRE(ExpressionTraits<identity_lift>::has_index_set);
}

TEST_CASE("lift_unit_index_set", "[expression][unary][lift][indexset]") {
    // Unit vector e_1 in R^4: nonzero only at index 1
    auto unit = expression::nullary::unit_vector<double, 4, 1>();
    using unit_type = std::decay_t<decltype(unit)>;

    // Lift: rank-1 → rank-2, dims = (4, dynamic)
    // Output dim 0 = child dim 0, dim 1 = lifted (broadcast)
    unary::Lift<1, const unit_type> rep(unit);

    REQUIRE(rep.extents().rank() == 2);

    SECTION("child dimension propagates Unit sparsity") {
        // index_set<0>(any_col) should give the Unit's nonzero index = {1}
        auto is0 = rep.index_set<0>(0);
        CHECK(is0.size() == 1);
        CHECK(is0.contains(1));
        CHECK_FALSE(is0.contains(0));
        CHECK_FALSE(is0.contains(2));
    }

    SECTION("lifted dimension returns FullRange") {
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

TEST_CASE("lift_identity_index_set", "[expression][unary][lift][indexset]") {
    expression::nullary::Identity<double, 3, 3> eye;
    using eye_type = std::decay_t<decltype(eye)>;

    // Lift with Count=2: rank-2 → rank-4
    // Output dims: (3, 3, dynamic, dynamic)
    // dim 0,1 = child dims, dim 2,3 = lifted
    unary::Lift<2, const eye_type> rep(eye);

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

    SECTION("lifted dimensions are fully dense") {
        auto is2 = rep.index_set<2>(0);
        CHECK(is2.contains(0));
        CHECK(is2.contains(42));

        auto is3 = rep.index_set<3>(0);
        CHECK(is3.contains(0));
        CHECK(is3.contains(42));
    }
}

TEST_CASE("lift_count0_identity_index_set", "[expression][unary][lift][indexset]") {
    // Count=0: identity-like, no dimensions added
    expression::nullary::Identity<double, 3, 3> eye;
    using eye_type = std::decay_t<decltype(eye)>;

    unary::Lift<0, const eye_type> rep(eye);

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

TEST_CASE("lift_count0_unit_rank1_index_set", "[expression][unary][lift][indexset]") {
    // Count=0 on rank-1: identity, output is still rank-1
    auto unit = expression::nullary::unit_vector<double, 5, 3>();
    using unit_type = std::decay_t<decltype(unit)>;

    unary::Lift<0, const unit_type> rep(unit);

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
