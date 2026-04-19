// Tests for the IndexSet infrastructure:
// - IndexSet concept satisfaction (static_asserts in header)
// - ContiguousIndexRange, SingleIndexRange, SparseIndexRange, FullRange
// - TriangularView::index_set<D>() for all 6 modes
// - Identity::index_set<D>() for rank-2
// - Unit::index_set<0>() for rank-1
// - HasIndexSet concept

#include <algorithm>
#include <ranges>
#include <vector>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/detail/ExpressionTraits.hpp>
#include <zipper/expression/detail/IndexSet.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/MDArray.hpp>
#include <zipper/expression/nullary/Unit.hpp>
#include <zipper/expression/unary/CoefficientWiseOperation.hpp>
#include <zipper/expression/unary/ScalarArithmetic.hpp>
#include <zipper/expression/unary/ScalarOperation.hpp>
#include <zipper/expression/unary/Slice.hpp>
#include <zipper/expression/unary/Swizzle.hpp>
#include <zipper/expression/unary/TriangularView.hpp>
#include <zipper/expression/unary/UnsafeRef.hpp>

#include "../catch_include.hpp"

using namespace zipper;
using namespace zipper::expression;
using namespace zipper::expression::detail;

// Helper to collect a range into a vector (iota_view iterators don't satisfy
// LegacyInputIterator so we can't use vector(begin, end) directly).
template <typename R>
static auto to_vec(const R &range) -> std::vector<index_type> {
    std::vector<index_type> result;
    for (auto it = range.begin(); it != range.end(); ++it) {
        result.push_back(*it);
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Range type unit tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("ContiguousIndexRange", "[index_set][range_types]") {
    SECTION("normal range [2, 5)") {
        ContiguousIndexRange r{2, 5};
        CHECK(!r.empty());
        CHECK(r.size() == 3);
        CHECK(!r.contains(0));
        CHECK(!r.contains(1));
        CHECK(r.contains(2));
        CHECK(r.contains(3));
        CHECK(r.contains(4));
        CHECK(!r.contains(5));

        auto collected = to_vec(r);
        REQUIRE(collected.size() == 3);
        CHECK(collected[0] == 2);
        CHECK(collected[1] == 3);
        CHECK(collected[2] == 4);
    }

    SECTION("empty range [3, 3)") {
        ContiguousIndexRange r{3, 3};
        CHECK(r.empty());
        CHECK(r.size() == 0);
        CHECK(!r.contains(3));
    }

    SECTION("single-element range [0, 1)") {
        ContiguousIndexRange r{0, 1};
        CHECK(!r.empty());
        CHECK(r.size() == 1);
        CHECK(r.contains(0));
        CHECK(!r.contains(1));
    }
}

TEST_CASE("SingleIndexRange", "[index_set][range_types]") {
    SingleIndexRange r{4};
    CHECK(!r.empty());
    CHECK(r.size() == 1);
    CHECK(!r.contains(3));
    CHECK(r.contains(4));
    CHECK(!r.contains(5));

    auto collected = to_vec(r);
    REQUIRE(collected.size() == 1);
    CHECK(collected[0] == 4);
}

TEST_CASE("SparseIndexRange", "[index_set][range_types]") {
    SparseIndexRange r{{1, 3, 7, 10}};
    CHECK(!r.empty());
    CHECK(r.size() == 4);
    CHECK(!r.contains(0));
    CHECK(r.contains(1));
    CHECK(!r.contains(2));
    CHECK(r.contains(3));
    CHECK(!r.contains(5));
    CHECK(r.contains(7));
    CHECK(r.contains(10));
    CHECK(!r.contains(11));

    // SparseIndexRange stores a vector, so its iterators are fine
    std::vector<index_type> collected(r.begin(), r.end());
    REQUIRE(collected.size() == 4);
    CHECK(collected[0] == 1);
    CHECK(collected[1] == 3);
    CHECK(collected[2] == 7);
    CHECK(collected[3] == 10);
}

TEST_CASE("FullRange", "[index_set][range_types]") {
    SECTION("extent 4") {
        FullRange r{4};
        CHECK(!r.empty());
        CHECK(r.size() == 4);
        CHECK(r.contains(0));
        CHECK(r.contains(3));
        // contains() always returns true
        CHECK(r.contains(100));

        auto collected = to_vec(r);
        REQUIRE(collected.size() == 4);
        CHECK(collected[0] == 0);
        CHECK(collected[3] == 3);
    }

    SECTION("extent 0") {
        FullRange r{0};
        CHECK(r.empty());
        CHECK(r.size() == 0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HasIndexSet concept checks
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("HasIndexSet concept", "[index_set][concept]") {
    // TriangularView has known zeros
    using IdentityExpr = nullary::Identity<double, 3, 3>;
    STATIC_CHECK(HasIndexSet<IdentityExpr>);

    using UnitExpr = nullary::Unit<double, 5, index_type>;
    STATIC_CHECK(HasIndexSet<UnitExpr>);

    // MDArray (dense) does NOT have known zeros
    using DenseExpr = nullary::MDArray<double, zipper::extents<3, 3>>;
    STATIC_CHECK_FALSE(HasIndexSet<DenseExpr>);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TriangularView index_set tests
// ═══════════════════════════════════════════════════════════════════════════════

static auto make_4x4() -> Matrix<double, 4, 4> {
    return Matrix<double, 4, 4>{
        {1.0, 2.0, 3.0, 4.0},
        {5.0, 6.0, 7.0, 8.0},
        {9.0, 10.0, 11.0, 12.0},
        {13.0, 14.0, 15.0, 16.0}};
}

TEST_CASE("TriangularView Lower index_set",
          "[index_set][triangular_view][lower]") {
    auto M = make_4x4();
    auto L = triangular_view<TriangularMode::Lower>(M);

    // index_set<1>(row) = col range = [0, row+1)
    SECTION("col_range_for_row") {
        auto r0 = L.col_range_for_row(0);
        CHECK(r0.first == 0);
        CHECK(r0.last == 1);
        CHECK(to_vec(r0) == std::vector<index_type>{0});

        auto r1 = L.col_range_for_row(1);
        CHECK(r1.first == 0);
        CHECK(r1.last == 2);
        CHECK(to_vec(r1) == std::vector<index_type>{0, 1});

        auto r3 = L.col_range_for_row(3);
        CHECK(r3.first == 0);
        CHECK(r3.last == 4);
        CHECK(r3.size() == 4);
    }

    // index_set<0>(col) = row range = [col, nrows)
    SECTION("row_range_for_col") {
        auto r0 = L.row_range_for_col(0);
        CHECK(r0.first == 0);
        CHECK(r0.last == 4);
        CHECK(r0.size() == 4);

        auto r3 = L.row_range_for_col(3);
        CHECK(r3.first == 3);
        CHECK(r3.last == 4);
        CHECK(to_vec(r3) == std::vector<index_type>{3});
    }
}

TEST_CASE("TriangularView StrictlyLower index_set",
          "[index_set][triangular_view][strictly_lower]") {
    auto M = make_4x4();
    auto SL = triangular_view<TriangularMode::StrictlyLower>(M);

    // index_set<1>(row) = [0, row)
    SECTION("col_range_for_row") {
        auto r0 = SL.col_range_for_row(0);
        CHECK(r0.empty());  // row 0 has no strictly-lower entries

        auto r1 = SL.col_range_for_row(1);
        CHECK(r1.first == 0);
        CHECK(r1.last == 1);

        auto r3 = SL.col_range_for_row(3);
        CHECK(r3.first == 0);
        CHECK(r3.last == 3);
    }

    // index_set<0>(col) = [col+1, nrows)
    SECTION("row_range_for_col") {
        auto r0 = SL.row_range_for_col(0);
        CHECK(r0.first == 1);
        CHECK(r0.last == 4);

        auto r3 = SL.row_range_for_col(3);
        CHECK(r3.empty());  // col 3 has no strictly-lower rows
    }
}

TEST_CASE("TriangularView UnitLower index_set",
          "[index_set][triangular_view][unit_lower]") {
    auto M = make_4x4();
    auto UL = triangular_view<TriangularMode::UnitLower>(M);

    // Same ranges as Lower: [0, row+1) for cols, [col, nrows) for rows
    auto r2 = UL.col_range_for_row(2);
    CHECK(r2.first == 0);
    CHECK(r2.last == 3);

    auto rc1 = UL.row_range_for_col(1);
    CHECK(rc1.first == 1);
    CHECK(rc1.last == 4);
}

TEST_CASE("TriangularView Upper index_set",
          "[index_set][triangular_view][upper]") {
    auto M = make_4x4();
    auto U = triangular_view<TriangularMode::Upper>(M);

    // index_set<1>(row) = [row, ncols)
    SECTION("col_range_for_row") {
        auto r0 = U.col_range_for_row(0);
        CHECK(r0.first == 0);
        CHECK(r0.last == 4);
        CHECK(r0.size() == 4);

        auto r3 = U.col_range_for_row(3);
        CHECK(r3.first == 3);
        CHECK(r3.last == 4);
    }

    // index_set<0>(col) = [0, col+1)
    SECTION("row_range_for_col") {
        auto r0 = U.row_range_for_col(0);
        CHECK(r0.first == 0);
        CHECK(r0.last == 1);

        auto r3 = U.row_range_for_col(3);
        CHECK(r3.first == 0);
        CHECK(r3.last == 4);
    }
}

TEST_CASE("TriangularView StrictlyUpper index_set",
          "[index_set][triangular_view][strictly_upper]") {
    auto M = make_4x4();
    auto SU = triangular_view<TriangularMode::StrictlyUpper>(M);

    // index_set<1>(row) = [row+1, ncols)
    SECTION("col_range_for_row") {
        auto r0 = SU.col_range_for_row(0);
        CHECK(r0.first == 1);
        CHECK(r0.last == 4);

        auto r3 = SU.col_range_for_row(3);
        CHECK(r3.empty());
    }

    // index_set<0>(col) = [0, col)
    SECTION("row_range_for_col") {
        auto r0 = SU.row_range_for_col(0);
        CHECK(r0.empty());

        auto r3 = SU.row_range_for_col(3);
        CHECK(r3.first == 0);
        CHECK(r3.last == 3);
    }
}

TEST_CASE("TriangularView UnitUpper index_set",
          "[index_set][triangular_view][unit_upper]") {
    auto M = make_4x4();
    auto UU = triangular_view<TriangularMode::UnitUpper>(M);

    // Same ranges as Upper: [row, ncols) for cols, [0, col+1) for rows
    auto r1 = UU.col_range_for_row(1);
    CHECK(r1.first == 1);
    CHECK(r1.last == 4);

    auto rc2 = UU.row_range_for_col(2);
    CHECK(rc2.first == 0);
    CHECK(rc2.last == 3);
}

// Verify that index_set results are consistent with coeff() values
TEST_CASE("TriangularView index_set consistency with coeff",
          "[index_set][triangular_view][consistency]") {
    auto M = make_4x4();
    auto L = triangular_view<TriangularMode::Lower>(M);

    for (index_type row = 0; row < 4; ++row) {
        auto col_range = L.col_range_for_row(row);
        for (index_type col = 0; col < 4; ++col) {
            if (!col_range.contains(col)) {
                // Outside the reported range => must be zero
                CHECK(L.coeff(row, col) == 0.0);
            }
        }
    }

    auto SU = triangular_view<TriangularMode::StrictlyUpper>(M);
    for (index_type col = 0; col < 4; ++col) {
        auto row_range = SU.row_range_for_col(col);
        for (index_type row = 0; row < 4; ++row) {
            if (!row_range.contains(row)) {
                CHECK(SU.coeff(row, col) == 0.0);
            }
        }
    }
}

// Verify that index_set returns use static_index_t<0> for zero-origin ranges
TEST_CASE("TriangularView index_set uses static first=0",
          "[index_set][triangular_view][static_zero]") {
    using CR0 = ContiguousIndexSet<zipper::static_index_t<0>, index_type>;

    auto M = make_4x4();

    SECTION("Lower col_range_for_row returns CR0") {
        auto L = triangular_view<TriangularMode::Lower>(M);
        auto r = L.col_range_for_row(2);
        STATIC_CHECK(std::is_same_v<decltype(r), CR0>);
        CHECK(r.first == 0);
        CHECK(r.last == 3);
    }

    SECTION("StrictlyLower col_range_for_row returns CR0") {
        auto SL = triangular_view<TriangularMode::StrictlyLower>(M);
        auto r = SL.col_range_for_row(3);
        STATIC_CHECK(std::is_same_v<decltype(r), CR0>);
        CHECK(r.first == 0);
        CHECK(r.last == 3);
    }

    SECTION("UnitLower col_range_for_row returns CR0") {
        auto UL = triangular_view<TriangularMode::UnitLower>(M);
        auto r = UL.col_range_for_row(2);
        STATIC_CHECK(std::is_same_v<decltype(r), CR0>);
        CHECK(r.first == 0);
        CHECK(r.last == 3);
    }

    SECTION("Upper row_range_for_col returns CR0") {
        auto U = triangular_view<TriangularMode::Upper>(M);
        auto r = U.row_range_for_col(3);
        STATIC_CHECK(std::is_same_v<decltype(r), CR0>);
        CHECK(r.first == 0);
        CHECK(r.last == 4);
    }

    SECTION("StrictlyUpper row_range_for_col returns CR0") {
        auto SU = triangular_view<TriangularMode::StrictlyUpper>(M);
        auto r = SU.row_range_for_col(3);
        STATIC_CHECK(std::is_same_v<decltype(r), CR0>);
        CHECK(r.first == 0);
        CHECK(r.last == 3);
    }

    SECTION("UnitUpper row_range_for_col returns CR0") {
        auto UU = triangular_view<TriangularMode::UnitUpper>(M);
        auto r = UU.row_range_for_col(2);
        STATIC_CHECK(std::is_same_v<decltype(r), CR0>);
        CHECK(r.first == 0);
        CHECK(r.last == 3);
    }

    SECTION("Upper col_range_for_row returns CR (not CR0)") {
        // Upper col range starts at row, not 0
        auto U = triangular_view<TriangularMode::Upper>(M);
        auto r = U.col_range_for_row(2);
        STATIC_CHECK(std::is_same_v<decltype(r), ContiguousIndexRange>);
        CHECK(r.first == 2);
        CHECK(r.last == 4);
    }

    SECTION("Lower row_range_for_col returns CR (not CR0)") {
        // Lower row range starts at col, not 0
        auto L = triangular_view<TriangularMode::Lower>(M);
        auto r = L.row_range_for_col(2);
        STATIC_CHECK(std::is_same_v<decltype(r), ContiguousIndexRange>);
        CHECK(r.first == 2);
        CHECK(r.last == 4);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Identity index_set tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Identity index_set rank-2",
          "[index_set][identity]") {
    nullary::Identity<double, 4, 4> I;

    SECTION("col_range_for_row returns SingleIndexRange") {
        for (index_type row = 0; row < 4; ++row) {
            auto r = I.col_range_for_row(row);
            CHECK(r.size() == 1);
            CHECK(r.value == row);
            CHECK(r.contains(row));
            CHECK(!r.contains(row == 0 ? 1 : 0));
        }
    }

    SECTION("row_range_for_col returns SingleIndexRange") {
        for (index_type col = 0; col < 4; ++col) {
            auto r = I.row_range_for_col(col);
            CHECK(r.size() == 1);
            CHECK(r.value == col);
        }
    }

    SECTION("consistency with coeff") {
        for (index_type row = 0; row < 4; ++row) {
            auto col_range = I.col_range_for_row(row);
            for (index_type col = 0; col < 4; ++col) {
                if (!col_range.contains(col)) {
                    CHECK(I.coeff(row, col) == 0.0);
                } else {
                    CHECK(I.coeff(row, col) == 1.0);
                }
            }
        }
    }
}

TEST_CASE("Identity index_set general rank",
          "[index_set][identity]") {
    // Rank-3 Identity: nonzero at (i,i,i)
    nullary::Identity<double, 3, 3, 3> I3;

    auto r0 = I3.index_set<0>(2);
    CHECK(r0.size() == 1);
    CHECK(r0.value == 2);

    auto r1 = I3.index_set<1>(1);
    CHECK(r1.size() == 1);
    CHECK(r1.value == 1);

    auto r2 = I3.index_set<2>(0);
    CHECK(r2.size() == 1);
    CHECK(r2.value == 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit index_set tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Unit index_set", "[index_set][unit]") {
    SECTION("static extent, dynamic index") {
        auto u = nullary::unit_vector<double, 5>(2);
        auto r = u.index_set<0>();
        CHECK(r.size() == 1);
        CHECK(r.value == 2);
        CHECK(r.contains(2));
        CHECK(!r.contains(0));
        CHECK(!r.contains(4));
    }

    SECTION("nonzero_segment convenience alias") {
        auto u = nullary::unit_vector<double, 5>(3);
        auto r = u.nonzero_segment();
        CHECK(r.value == 3);
    }

    SECTION("static extent, static index") {
        auto u = nullary::unit_vector<double, 5, 1>();
        auto r = u.index_set<0>();
        CHECK(r.value == 1);
    }

    SECTION("dynamic extent") {
        auto u = nullary::unit_vector<double>(8, 5);
        auto r = u.nonzero_segment();
        CHECK(r.value == 5);
    }

    SECTION("consistency with coeff") {
        auto u = nullary::unit_vector<double, 5>(2);
        auto r = u.nonzero_segment();
        for (index_type i = 0; i < 5; ++i) {
            if (!r.contains(i)) {
                CHECK(u.coeff(i) == 0.0);
            } else {
                CHECK(u.coeff(i) == 1.0);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Zero-preserving operation trait tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("ZeroPreservingUnaryOp concept", "[index_set][traits]") {
    using namespace zipper::expression::detail;

    // negate is zero-preserving
    STATIC_CHECK(ZeroPreservingUnaryOp<std::negate<double>>);
    STATIC_CHECK(ZeroPreservingUnaryOp<std::negate<int>>);

    // logical_not is NOT zero-preserving: !0 == true
    STATIC_CHECK_FALSE(ZeroPreservingUnaryOp<std::logical_not<double>>);

    // bit_not is NOT zero-preserving: ~0 != 0
    STATIC_CHECK_FALSE(ZeroPreservingUnaryOp<std::bit_not<int>>);
}

TEST_CASE("ZeroPreservingScalarOp concept", "[index_set][traits]") {
    using namespace zipper::expression::detail;

    // multiplies: 0*s == 0 and s*0 == 0
    STATIC_CHECK(ZeroPreservingScalarOp<std::multiplies<double>, true>);
    STATIC_CHECK(ZeroPreservingScalarOp<std::multiplies<double>, false>);

    // divides: 0/s == 0 (scalar on right only)
    STATIC_CHECK(ZeroPreservingScalarOp<std::divides<double>, true>);
    STATIC_CHECK_FALSE(ZeroPreservingScalarOp<std::divides<double>, false>);

    // plus: 0+s != 0 in general
    STATIC_CHECK_FALSE(ZeroPreservingScalarOp<std::plus<double>, true>);
    STATIC_CHECK_FALSE(ZeroPreservingScalarOp<std::plus<double>, false>);

    // minus: 0-s != 0 in general
    STATIC_CHECK_FALSE(ZeroPreservingScalarOp<std::minus<double>, true>);
    STATIC_CHECK_FALSE(ZeroPreservingScalarOp<std::minus<double>, false>);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Transitivity: UnsafeRef
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("UnsafeRef propagates has_index_set",
          "[index_set][transitivity][unsafe_ref]") {
    auto M = make_4x4();
    auto L = triangular_view<TriangularMode::Lower>(M);

    // UnsafeRef wrapping a TriangularView should have known zeros
    auto uref = unary::UnsafeRef(L);
    using uref_type = decltype(uref);
    using uref_traits = zipper::expression::detail::ExpressionTraits<
        std::decay_t<uref_type>>;
    STATIC_CHECK(uref_traits::has_index_set);

    SECTION("col_range_for_row matches child") {
        for (index_type row = 0; row < 4; ++row) {
            auto ur = uref.col_range_for_row(row);
            auto lr = L.col_range_for_row(row);
            CHECK(ur.first == lr.first);
            CHECK(ur.last == lr.last);
        }
    }

    SECTION("row_range_for_col matches child") {
        for (index_type col = 0; col < 4; ++col) {
            auto ur = uref.row_range_for_col(col);
            auto lr = L.row_range_for_col(col);
            CHECK(ur.first == lr.first);
            CHECK(ur.last == lr.last);
        }
    }

    SECTION("UnsafeRef on dense expression does NOT have known zeros") {
        using DenseExpr = nullary::MDArray<double, zipper::extents<3, 3>>;
        using URefDense = unary::UnsafeRef<const DenseExpr &>;
        using dense_traits = zipper::expression::detail::ExpressionTraits<URefDense>;
        STATIC_CHECK_FALSE(dense_traits::has_index_set);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Transitivity: Swizzle (Transpose)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Swizzle/Transpose propagates has_index_set",
          "[index_set][transitivity][swizzle]") {
    auto M = make_4x4();
    auto L = triangular_view<TriangularMode::Lower>(M);

    // Transpose of Lower = Upper (structurally)
    auto LT = unary::Swizzle<decltype(L), 1, 0>(L);
    using lt_type = decltype(LT);
    using lt_traits = zipper::expression::detail::ExpressionTraits<
        std::decay_t<lt_type>>;
    STATIC_CHECK(lt_traits::has_index_set);

    SECTION("transposed col_range_for_row swaps to row_range_for_col") {
        // LT.col_range_for_row(r) should == L.row_range_for_col(r)
        // because transposing swaps dimensions
        for (index_type r = 0; r < 4; ++r) {
            auto lt_cr = LT.col_range_for_row(r);
            auto l_rr = L.row_range_for_col(r);
            CHECK(lt_cr.first == l_rr.first);
            CHECK(lt_cr.last == l_rr.last);
        }
    }

    SECTION("transposed row_range_for_col swaps to col_range_for_row") {
        for (index_type c = 0; c < 4; ++c) {
            auto lt_rr = LT.row_range_for_col(c);
            auto l_cr = L.col_range_for_row(c);
            CHECK(lt_rr.first == l_cr.first);
            CHECK(lt_rr.last == l_cr.last);
        }
    }

    SECTION("consistency with coeff values") {
        for (index_type row = 0; row < 4; ++row) {
            auto col_range = LT.col_range_for_row(row);
            for (index_type col = 0; col < 4; ++col) {
                if (!col_range.contains(col)) {
                    CHECK(LT.coeff(row, col) == 0.0);
                }
            }
        }
    }

    SECTION("Swizzle on dense has no known zeros") {
        using DenseExpr = nullary::MDArray<double, zipper::extents<3, 3>>;
        using SwDense = unary::Swizzle<const DenseExpr &, 1, 0>;
        using sw_traits = zipper::expression::detail::ExpressionTraits<SwDense>;
        STATIC_CHECK_FALSE(sw_traits::has_index_set);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Transitivity: CoefficientWiseOperation (Negate)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Negate propagates has_index_set",
          "[index_set][transitivity][cwiseop]") {
    auto M = make_4x4();
    auto U = triangular_view<TriangularMode::Upper>(M);

    // Negate of TriangularView should preserve known zeros
    auto negU = unary::CoefficientWiseOperation<decltype(U), std::negate<double>>(U);
    using neg_type = decltype(negU);
    using neg_traits = zipper::expression::detail::ExpressionTraits<
        std::decay_t<neg_type>>;
    STATIC_CHECK(neg_traits::has_index_set);

    SECTION("ranges match child") {
        for (index_type row = 0; row < 4; ++row) {
            auto nr = negU.col_range_for_row(row);
            auto ur = U.col_range_for_row(row);
            CHECK(nr.first == ur.first);
            CHECK(nr.last == ur.last);
        }
    }

    SECTION("consistency with coeff") {
        for (index_type row = 0; row < 4; ++row) {
            auto col_range = negU.col_range_for_row(row);
            for (index_type col = 0; col < 4; ++col) {
                if (!col_range.contains(col)) {
                    CHECK(negU.coeff(row, col) == 0.0);
                }
            }
        }
    }

    SECTION("LogicalNot does NOT propagate") {
        using LNotType = unary::LogicalNot<decltype(U)>;
        using lnot_traits = zipper::expression::detail::ExpressionTraits<
            std::decay_t<LNotType>>;
        STATIC_CHECK_FALSE(lnot_traits::has_index_set);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Transitivity: ScalarOperation (ScalarMultiplies, ScalarDivides)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("ScalarMultiplies propagates has_index_set",
          "[index_set][transitivity][scalarop]") {
    auto M = make_4x4();
    auto L = triangular_view<TriangularMode::Lower>(M);

    // expr * scalar (ScalarOnRight = true)
    using MulRight = unary::ScalarMultiplies<double, decltype(L), true>;
    using mul_right_traits = zipper::expression::detail::ExpressionTraits<
        std::decay_t<MulRight>>;
    STATIC_CHECK(mul_right_traits::has_index_set);

    // scalar * expr (ScalarOnRight = false)
    using MulLeft = unary::ScalarMultiplies<double, decltype(L), false>;
    using mul_left_traits = zipper::expression::detail::ExpressionTraits<
        std::decay_t<MulLeft>>;
    STATIC_CHECK(mul_left_traits::has_index_set);

    auto scaled = MulRight(L, 3.0);
    SECTION("ranges match child after scaling") {
        for (index_type row = 0; row < 4; ++row) {
            auto sr = scaled.col_range_for_row(row);
            auto lr = L.col_range_for_row(row);
            CHECK(sr.first == lr.first);
            CHECK(sr.last == lr.last);
        }
    }
}

TEST_CASE("ScalarDivides propagates only when ScalarOnRight",
          "[index_set][transitivity][scalarop]") {
    auto M = make_4x4();
    auto L = triangular_view<TriangularMode::Lower>(M);

    // expr / scalar (ScalarOnRight = true): 0/s == 0
    using DivRight = unary::ScalarDivides<double, decltype(L), true>;
    using div_right_traits = zipper::expression::detail::ExpressionTraits<
        std::decay_t<DivRight>>;
    STATIC_CHECK(div_right_traits::has_index_set);

    // scalar / expr (ScalarOnRight = false): s/0 is undefined, not zero
    using DivLeft = unary::ScalarDivides<double, decltype(L), false>;
    using div_left_traits = zipper::expression::detail::ExpressionTraits<
        std::decay_t<DivLeft>>;
    STATIC_CHECK_FALSE(div_left_traits::has_index_set);
}

TEST_CASE("ScalarPlus does NOT propagate has_index_set",
          "[index_set][transitivity][scalarop]") {
    auto M = make_4x4();
    auto L = triangular_view<TriangularMode::Lower>(M);

    // expr + scalar: 0 + s != 0
    using PlusRight = unary::ScalarPlus<double, decltype(L), true>;
    using plus_traits = zipper::expression::detail::ExpressionTraits<
        std::decay_t<PlusRight>>;
    STATIC_CHECK_FALSE(plus_traits::has_index_set);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Transitivity: Slice on TriangularView
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Slice propagates has_index_set from TriangularView",
          "[index_set][transitivity][slice]") {
    auto M = make_4x4();
    auto L = triangular_view<TriangularMode::Lower>(M);

    SECTION("full_extent slice preserves ranges exactly") {
        // Slice taking all rows, all cols = identity
        auto sl = unary::Slice(L, full_extent_t{}, full_extent_t{});
        using sl_type = decltype(sl);
        using sl_traits = zipper::expression::detail::ExpressionTraits<
            std::decay_t<sl_type>>;
        STATIC_CHECK(sl_traits::has_index_set);

        for (index_type row = 0; row < 4; ++row) {
            auto sr = sl.col_range_for_row(row);
            auto lr = L.col_range_for_row(row);
            CHECK(sr.first == lr.first);
            CHECK(sr.last == lr.last);
        }
    }

    SECTION("row extraction: Slice with integer + full_extent") {
        // Extract row 2 from Lower triangular: [L(2,0), L(2,1), L(2,2), 0]
        // This produces a rank-1 output (row is collapsed).
        // has_index_set is true because child has it.
        auto row_slice = unary::Slice(L, index_type{2}, full_extent_t{});
        using rs_type = decltype(row_slice);
        using rs_traits = zipper::expression::detail::ExpressionTraits<
            std::decay_t<rs_type>>;
        STATIC_CHECK(rs_traits::has_index_set);

        // For a rank-1 slice, index_set<0>() queries along
        // the only output dimension (cols). The child dimension for
        // "cols" is 1, and the fixed index is row=2.
        // L.index_set<1>(2) = [0, 3), so the output should be [0, 3).
        auto r = row_slice.template index_set<0>(2);
        CHECK(r.first == 0);
        CHECK(r.last == 3);

        // Verify consistency: outside range should be zero
        for (index_type col = 0; col < 4; ++col) {
            if (!r.contains(col)) {
                CHECK(row_slice.coeff(col) == 0.0);
            }
        }
    }

    SECTION("sub-block slice with strided_slice") {
        // Take rows [1, 3) and cols [0, 4) from Lower
        auto sub = unary::Slice(L,
            strided_slice{index_type{1}, index_type{2}, index_type{1}},
            full_extent_t{});
        using sub_type = decltype(sub);
        using sub_traits = zipper::expression::detail::ExpressionTraits<
            std::decay_t<sub_type>>;
        STATIC_CHECK(sub_traits::has_index_set);

        // Output row 0 maps to child row 1.
        // L.index_set<1>(1) = [0, 2)
        // Remap through full_extent on cols: [0, 2)
        auto r0 = sub.col_range_for_row(0);
        CHECK(r0.first == 0);
        CHECK(r0.last == 2);

        // Output row 1 maps to child row 2.
        // L.index_set<1>(2) = [0, 3)
        auto r1 = sub.col_range_for_row(1);
        CHECK(r1.first == 0);
        CHECK(r1.last == 3);

        // Verify consistency
        for (index_type row = 0; row < 2; ++row) {
            auto cr = sub.col_range_for_row(row);
            for (index_type col = 0; col < 4; ++col) {
                if (!cr.contains(col)) {
                    CHECK(sub.coeff(row, col) == 0.0);
                }
            }
        }
    }

    SECTION("col-restricted sub-block") {
        // Take all rows, cols [1, 3) from Upper
        auto U = triangular_view<TriangularMode::Upper>(M);
        auto sub = unary::Slice(U,
            full_extent_t{},
            strided_slice{index_type{1}, index_type{2}, index_type{1}});
        using sub_type = decltype(sub);
        using sub_traits = zipper::expression::detail::ExpressionTraits<
            std::decay_t<sub_type>>;
        STATIC_CHECK(sub_traits::has_index_set);

        // Output col 0 maps to child col 1.
        // U.index_set<0>(1) = [0, 2) (rows with non-zeros in child col 1)
        // Remap through full_extent on rows: [0, 2)
        auto r0 = sub.row_range_for_col(0);
        CHECK(r0.first == 0);
        CHECK(r0.last == 2);

        // Output col 1 maps to child col 2.
        // U.index_set<0>(2) = [0, 3)
        auto r1 = sub.row_range_for_col(1);
        CHECK(r1.first == 0);
        CHECK(r1.last == 3);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Transitivity: chained operations
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Chained transitivity: Negate(Transpose(TriangularView))",
          "[index_set][transitivity][chained]") {
    auto M = make_4x4();
    auto L = triangular_view<TriangularMode::Lower>(M);

    // Transpose(Lower) -> structurally Upper
    auto LT = unary::Swizzle<decltype(L), 1, 0>(L);

    // Negate(Transpose(Lower))
    auto negLT = unary::CoefficientWiseOperation<decltype(LT), std::negate<double>>(LT);

    using neg_traits = zipper::expression::detail::ExpressionTraits<
        std::decay_t<decltype(negLT)>>;
    STATIC_CHECK(neg_traits::has_index_set);

    // The col_range_for_row should match Upper pattern
    SECTION("ranges match Upper pattern") {
        for (index_type row = 0; row < 4; ++row) {
            auto r = negLT.col_range_for_row(row);
            // Transpose of Lower gives Upper: col range = [row, 4)
            CHECK(r.first == row);
            CHECK(r.last == 4);
        }
    }

    SECTION("consistency with coeff") {
        for (index_type row = 0; row < 4; ++row) {
            auto col_range = negLT.col_range_for_row(row);
            for (index_type col = 0; col < 4; ++col) {
                if (!col_range.contains(col)) {
                    CHECK(negLT.coeff(row, col) == 0.0);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DisjointRange unit tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("DisjointRange: basic construction and properties",
          "[index_set][disjoint_range]") {
    using CR = ContiguousIndexRange;

    SECTION("two non-overlapping segments") {
        DisjointRange<CR, CR> dr{std::tuple{CR{1, 3}, CR{5, 8}}};
        CHECK(dr.size() == 5);    // [1,2] + [5,6,7]
        CHECK_FALSE(dr.empty());
        CHECK(dr.num_segments == 2);
    }

    SECTION("one segment empty") {
        DisjointRange<CR, CR> dr{std::tuple{CR{0, 0}, CR{2, 4}}};
        CHECK(dr.size() == 2);
        CHECK_FALSE(dr.empty());
    }

    SECTION("both segments empty") {
        DisjointRange<CR, CR> dr{std::tuple{CR{0, 0}, CR{0, 0}}};
        CHECK(dr.size() == 0);
        CHECK(dr.empty());
    }

    SECTION("single segment") {
        DisjointRange<CR> dr{std::tuple{CR{3, 7}}};
        CHECK(dr.size() == 4);
        CHECK_FALSE(dr.empty());
    }
}

TEST_CASE("DisjointRange: contains",
          "[index_set][disjoint_range]") {
    using CR = ContiguousIndexRange;
    DisjointRange<CR, CR> dr{std::tuple{CR{1, 3}, CR{5, 8}}};

    // In first segment
    CHECK(dr.contains(1));
    CHECK(dr.contains(2));

    // Between segments (gap)
    CHECK_FALSE(dr.contains(3));
    CHECK_FALSE(dr.contains(4));

    // In second segment
    CHECK(dr.contains(5));
    CHECK(dr.contains(6));
    CHECK(dr.contains(7));

    // Outside
    CHECK_FALSE(dr.contains(0));
    CHECK_FALSE(dr.contains(8));
}

TEST_CASE("DisjointRange: begin/end iteration",
          "[index_set][disjoint_range]") {
    using CR = ContiguousIndexRange;

    SECTION("two segments") {
        DisjointRange<CR, CR> dr{std::tuple{CR{1, 3}, CR{5, 8}}};
        auto v = to_vec(dr);
        REQUIRE(v.size() == 5);
        CHECK(v[0] == 1);
        CHECK(v[1] == 2);
        CHECK(v[2] == 5);
        CHECK(v[3] == 6);
        CHECK(v[4] == 7);
    }

    SECTION("first segment empty, second non-empty") {
        DisjointRange<CR, CR> dr{std::tuple{CR{0, 0}, CR{2, 4}}};
        auto v = to_vec(dr);
        REQUIRE(v.size() == 2);
        CHECK(v[0] == 2);
        CHECK(v[1] == 3);
    }

    SECTION("both empty") {
        DisjointRange<CR, CR> dr{std::tuple{CR{0, 0}, CR{0, 0}}};
        auto v = to_vec(dr);
        CHECK(v.empty());
    }

    SECTION("single element segments") {
        DisjointRange<CR, CR> dr{std::tuple{CR{0, 1}, CR{9, 10}}};
        auto v = to_vec(dr);
        REQUIRE(v.size() == 2);
        CHECK(v[0] == 0);
        CHECK(v[1] == 9);
    }
}

TEST_CASE("DisjointRange: for_each compile-time unrolled iteration",
          "[index_set][disjoint_range]") {
    using CR = ContiguousIndexRange;
    DisjointRange<CR, CR> dr{std::tuple{CR{1, 3}, CR{5, 8}}};

    std::vector<index_type> collected;
    dr.for_each([&](index_type idx) { collected.push_back(idx); });

    REQUIRE(collected.size() == 5);
    CHECK(collected[0] == 1);
    CHECK(collected[1] == 2);
    CHECK(collected[2] == 5);
    CHECK(collected[3] == 6);
    CHECK(collected[4] == 7);
}

TEST_CASE("DisjointRange: HasForEach concept satisfied",
          "[index_set][disjoint_range]") {
    using CR = ContiguousIndexRange;
    STATIC_CHECK(HasForEach<DisjointRange<CR, CR>>);
    STATIC_CHECK(HasForEach<DisjointRange<CR>>);

    // ContiguousIndexRange itself does NOT have for_each
    STATIC_CHECK_FALSE(HasForEach<CR>);
    STATIC_CHECK_FALSE(HasForEach<SingleIndexRange>);
    STATIC_CHECK_FALSE(HasForEach<FullRange>);
}

TEST_CASE("DisjointRange: to_contiguous_range bounding box",
          "[index_set][disjoint_range]") {
    using CR = ContiguousIndexRange;

    SECTION("two segments") {
        DisjointRange<CR, CR> dr{std::tuple{CR{2, 4}, CR{7, 10}}};
        auto bb = to_contiguous_range(dr);
        CHECK(bb.first == 2);
        CHECK(bb.last == 10);
    }

    SECTION("single segment") {
        DisjointRange<CR> dr{std::tuple{CR{3, 6}}};
        auto bb = to_contiguous_range(dr);
        CHECK(bb.first == 3);
        CHECK(bb.last == 6);
    }

    SECTION("all empty") {
        DisjointRange<CR, CR> dr{std::tuple{CR{0, 0}, CR{0, 0}}};
        auto bb = to_contiguous_range(dr);
        CHECK(bb.first == 0);
        CHECK(bb.last == 0);
    }

    SECTION("first empty, second non-empty") {
        DisjointRange<CR, CR> dr{std::tuple{CR{0, 0}, CR{5, 8}}};
        auto bb = to_contiguous_range(dr);
        CHECK(bb.first == 5);
        CHECK(bb.last == 8);
    }
}

TEST_CASE("DisjointRange: IndexSet concept satisfied",
          "[index_set][disjoint_range]") {
    using CR = ContiguousIndexRange;
    STATIC_CHECK(IndexSet<DisjointRange<CR>>);
    STATIC_CHECK(IndexSet<DisjointRange<CR, CR>>);
    STATIC_CHECK(IndexSet<DisjointRange<CR, CR, CR>>);
}

TEST_CASE("DisjointRange: IsDisjointRange concept",
          "[index_set][disjoint_range]") {
    using CR = ContiguousIndexRange;
    STATIC_CHECK(IsDisjointRange<DisjointRange<CR>>);
    STATIC_CHECK(IsDisjointRange<DisjointRange<CR, CR>>);
    STATIC_CHECK_FALSE(IsDisjointRange<CR>);
    STATIC_CHECK_FALSE(IsDisjointRange<SingleIndexRange>);
    STATIC_CHECK_FALSE(IsDisjointRange<FullRange>);
}

// ═══════════════════════════════════════════════════════════════════════════════
// range_union tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("range_union: two non-overlapping CRs",
          "[index_set][range_union]") {
    using CR = ContiguousIndexRange;
    auto result = range_union(CR{1, 3}, CR{5, 8});
    STATIC_CHECK(IsDisjointRange<decltype(result)>);

    // Should have two distinct segments
    auto bb = to_contiguous_range(result);
    CHECK(bb.first == 1);
    CHECK(bb.last == 8);

    // Check actual indices via iteration
    auto v = to_vec(result);
    REQUIRE(v.size() == 5);
    CHECK(v[0] == 1);
    CHECK(v[1] == 2);
    CHECK(v[2] == 5);
    CHECK(v[3] == 6);
    CHECK(v[4] == 7);

    // Gap indices not present
    CHECK_FALSE(result.contains(3));
    CHECK_FALSE(result.contains(4));
}

TEST_CASE("range_union: two overlapping CRs merge into one",
          "[index_set][range_union]") {
    using CR = ContiguousIndexRange;
    auto result = range_union(CR{1, 5}, CR{3, 8});

    // Should merge to [1, 8), one active segment + one empty
    auto bb = to_contiguous_range(result);
    CHECK(bb.first == 1);
    CHECK(bb.last == 8);
    CHECK(result.size() == 7);
}

TEST_CASE("range_union: two adjacent CRs merge",
          "[index_set][range_union]") {
    using CR = ContiguousIndexRange;
    auto result = range_union(CR{1, 4}, CR{4, 7});

    auto bb = to_contiguous_range(result);
    CHECK(bb.first == 1);
    CHECK(bb.last == 7);
    CHECK(result.size() == 6);
}

TEST_CASE("range_union: one empty CR",
          "[index_set][range_union]") {
    using CR = ContiguousIndexRange;
    auto result = range_union(CR{0, 0}, CR{3, 6});

    auto bb = to_contiguous_range(result);
    CHECK(bb.first == 3);
    CHECK(bb.last == 6);
    CHECK(result.size() == 3);
}

TEST_CASE("range_union: both empty CRs",
          "[index_set][range_union]") {
    using CR = ContiguousIndexRange;
    auto result = range_union(CR{0, 0}, CR{0, 0});

    CHECK(result.empty());
    CHECK(result.size() == 0);
}

TEST_CASE("range_union: SingleIndexRange + CR promotes",
          "[index_set][range_union]") {
    using CR = ContiguousIndexRange;
    auto result = range_union(SingleIndexRange{5}, CR{8, 10});
    STATIC_CHECK(IsDisjointRange<decltype(result)>);

    CHECK(result.contains(5));
    CHECK_FALSE(result.contains(6));
    CHECK_FALSE(result.contains(7));
    CHECK(result.contains(8));
    CHECK(result.contains(9));
    CHECK(result.size() == 3);
}

TEST_CASE("range_union: DR + CR",
          "[index_set][range_union]") {
    using CR = ContiguousIndexRange;
    DisjointRange<CR, CR> dr{std::tuple{CR{0, 2}, CR{5, 7}}};

    auto result = range_union(dr, CR{3, 4});
    STATIC_CHECK(IsDisjointRange<decltype(result)>);

    // Should have three non-empty segments: [0,2), [3,4), [5,7)
    CHECK(result.contains(0));
    CHECK(result.contains(1));
    CHECK_FALSE(result.contains(2));
    CHECK(result.contains(3));
    CHECK_FALSE(result.contains(4));
    CHECK(result.contains(5));
    CHECK(result.contains(6));
    CHECK(result.size() == 5);
}

TEST_CASE("range_union: DR + DR",
          "[index_set][range_union]") {
    using CR = ContiguousIndexRange;
    DisjointRange<CR, CR> dr1{std::tuple{CR{0, 2}, CR{6, 8}}};
    DisjointRange<CR, CR> dr2{std::tuple{CR{3, 5}, CR{7, 10}}};

    auto result = range_union(dr1, dr2);
    STATIC_CHECK(IsDisjointRange<decltype(result)>);

    // Expected: [0,2), [3,5), [6,10)
    CHECK(result.contains(0));
    CHECK(result.contains(1));
    CHECK_FALSE(result.contains(2));
    CHECK(result.contains(3));
    CHECK(result.contains(4));
    CHECK_FALSE(result.contains(5));
    CHECK(result.contains(6));
    CHECK(result.contains(7));
    CHECK(result.contains(8));
    CHECK(result.contains(9));
    CHECK(result.size() == 8);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TriangularMode::OffDiagonal tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("OffDiagonal: is valid triangular mode",
          "[index_set][off_diagonal]") {
    STATIC_CHECK(is_valid_triangular_mode<TriangularMode::OffDiagonal>());
}

TEST_CASE("OffDiagonal: coeff returns zero on diagonal, child elsewhere",
          "[index_set][off_diagonal]") {
    auto M = make_4x4();
    auto OD = triangular_view<TriangularMode::OffDiagonal>(M);

    SECTION("diagonal is zero") {
        for (index_type i = 0; i < 4; ++i) {
            CHECK(OD.coeff(i, i) == 0.0);
        }
    }

    SECTION("off-diagonal matches child") {
        for (index_type i = 0; i < 4; ++i) {
            for (index_type j = 0; j < 4; ++j) {
                if (i != j) {
                    CHECK(OD.coeff(i, j) == Catch::Approx(M(i, j)));
                }
            }
        }
    }
}

TEST_CASE("OffDiagonal: index_set returns DisjointRange",
          "[index_set][off_diagonal]") {
    auto M = make_4x4();
    auto OD = triangular_view<TriangularMode::OffDiagonal>(M);

    SECTION("col_range_for_row returns DisjointRange<CR,CR>") {
        // Row 0: [0,0) ∪ [1,4) → empty first seg, second = [1,4)
        auto r0 = OD.col_range_for_row(0);
        STATIC_CHECK(IsDisjointRange<decltype(r0)>);
        CHECK_FALSE(r0.contains(0));
        CHECK(r0.contains(1));
        CHECK(r0.contains(2));
        CHECK(r0.contains(3));
        CHECK(r0.size() == 3);

        // Row 2: [0,2) ∪ [3,4)
        auto r2 = OD.col_range_for_row(2);
        CHECK(r2.contains(0));
        CHECK(r2.contains(1));
        CHECK_FALSE(r2.contains(2));
        CHECK(r2.contains(3));
        CHECK(r2.size() == 3);

        // Row 3: [0,3) ∪ [4,4) → first = [0,3), second empty
        auto r3 = OD.col_range_for_row(3);
        CHECK(r3.contains(0));
        CHECK(r3.contains(1));
        CHECK(r3.contains(2));
        CHECK_FALSE(r3.contains(3));
        CHECK(r3.size() == 3);
    }

    SECTION("row_range_for_col returns DisjointRange<CR,CR>") {
        // Col 0: [0,0) ∪ [1,4) → [1,4)
        auto c0 = OD.row_range_for_col(0);
        STATIC_CHECK(IsDisjointRange<decltype(c0)>);
        CHECK_FALSE(c0.contains(0));
        CHECK(c0.contains(1));
        CHECK(c0.contains(2));
        CHECK(c0.contains(3));

        // Col 1: [0,1) ∪ [2,4)
        auto c1 = OD.row_range_for_col(1);
        CHECK(c1.contains(0));
        CHECK_FALSE(c1.contains(1));
        CHECK(c1.contains(2));
        CHECK(c1.contains(3));
    }
}

TEST_CASE("OffDiagonal: index_set consistency with coeff",
          "[index_set][off_diagonal]") {
    auto M = make_4x4();
    auto OD = triangular_view<TriangularMode::OffDiagonal>(M);

    for (index_type row = 0; row < 4; ++row) {
        auto col_range = OD.col_range_for_row(row);
        for (index_type col = 0; col < 4; ++col) {
            if (!col_range.contains(col)) {
                CHECK(OD.coeff(row, col) == 0.0);
            }
        }
    }

    for (index_type col = 0; col < 4; ++col) {
        auto row_range = OD.row_range_for_col(col);
        for (index_type row = 0; row < 4; ++row) {
            if (!row_range.contains(row)) {
                CHECK(OD.coeff(row, col) == 0.0);
            }
        }
    }
}

TEST_CASE("OffDiagonal: for_each on index_set",
          "[index_set][off_diagonal]") {
    auto M = make_4x4();
    auto OD = triangular_view<TriangularMode::OffDiagonal>(M);

    // for_each should visit exactly the off-diagonal columns for a given row
    for (index_type row = 0; row < 4; ++row) {
        auto col_range = OD.col_range_for_row(row);
        std::vector<index_type> visited;
        col_range.for_each([&](index_type col) { visited.push_back(col); });

        // Should have exactly 3 entries (all except the diagonal)
        REQUIRE(visited.size() == 3);
        for (auto col : visited) {
            CHECK(col != row);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Slice + DisjointRange tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Slice of OffDiagonal: full_extent preserves DisjointRange",
          "[index_set][slice][off_diagonal]") {
    auto M = make_4x4();
    auto OD = triangular_view<TriangularMode::OffDiagonal>(M);

    // Full extent slice = identity transformation
    auto sl = unary::Slice(OD, full_extent_t{}, full_extent_t{});
    using sl_traits = zipper::expression::detail::ExpressionTraits<
        std::decay_t<decltype(sl)>>;
    STATIC_CHECK(sl_traits::has_index_set);

    for (index_type row = 0; row < 4; ++row) {
        auto sr = sl.col_range_for_row(row);
        auto or_ = OD.col_range_for_row(row);

        // Both should produce the same set of indices
        auto sv = to_vec(sr);
        auto ov = to_vec(or_);
        CHECK(sv == ov);
    }

    // Verify consistency: outside range → zero
    for (index_type row = 0; row < 4; ++row) {
        auto col_range = sl.col_range_for_row(row);
        for (index_type col = 0; col < 4; ++col) {
            if (!col_range.contains(col)) {
                CHECK(sl.coeff(row, col) == 0.0);
            }
        }
    }
}

TEST_CASE("Slice of OffDiagonal: row extraction preserves disjoint structure",
          "[index_set][slice][off_diagonal]") {
    auto M = make_4x4();
    auto OD = triangular_view<TriangularMode::OffDiagonal>(M);

    // Extract row 2: OD(2, :)
    // OffDiagonal index_set<1>(2) = [0,2) ∪ [3,4) (DisjointRange)
    // After row extraction, the output is rank-1. The index_set<0>
    // should reflect the same disjoint structure (or its bounding box
    // through _extract_contiguous).
    auto row_slice = unary::Slice(OD, index_type{2}, full_extent_t{});
    using rs_traits = zipper::expression::detail::ExpressionTraits<
        std::decay_t<decltype(row_slice)>>;
    STATIC_CHECK(rs_traits::has_index_set);

    // Verify coefficient correctness
    for (index_type col = 0; col < 4; ++col) {
        double expected = (col == 2) ? 0.0 : M(2, col);
        CHECK(row_slice.coeff(col) == Catch::Approx(expected));
    }
}

TEST_CASE("Slice of OffDiagonal: sub-block slice",
          "[index_set][slice][off_diagonal]") {
    auto M = make_4x4();
    auto OD = triangular_view<TriangularMode::OffDiagonal>(M);

    // Take rows [1,3) and all cols from OffDiagonal
    auto sub = unary::Slice(OD,
        strided_slice{index_type{1}, index_type{2}, index_type{1}},
        full_extent_t{});
    using sub_traits = zipper::expression::detail::ExpressionTraits<
        std::decay_t<decltype(sub)>>;
    STATIC_CHECK(sub_traits::has_index_set);

    // Output row 0 = child row 1. OffDiag col range for row 1 = [0,1) ∪ [2,4)
    // Output row 1 = child row 2. OffDiag col range for row 2 = [0,2) ∪ [3,4)

    // Verify coefficient correctness
    for (index_type r = 0; r < 2; ++r) {
        index_type child_row = r + 1;
        for (index_type c = 0; c < 4; ++c) {
            double expected = (child_row == c) ? 0.0 : M(child_row, c);
            CHECK(sub.coeff(r, c) == Catch::Approx(expected));
        }
    }

    // Verify range consistency: outside reported range → zero
    for (index_type r = 0; r < 2; ++r) {
        auto col_range = sub.col_range_for_row(r);
        for (index_type c = 0; c < 4; ++c) {
            if (!col_range.contains(c)) {
                CHECK(sub.coeff(r, c) == 0.0);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 1 tests: StridedIndexSet
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("StridedIndexSet: runtime stride",
          "[index_set][strided_index_set]") {
    SECTION("stride 2 over [0, 10)") {
        StridedIndexSet<index_type, index_type, index_type> r{0, 10, 2};
        CHECK(!r.empty());
        CHECK(r.size() == 5);
        CHECK(r.contains(0));
        CHECK(!r.contains(1));
        CHECK(r.contains(2));
        CHECK(!r.contains(3));
        CHECK(r.contains(4));
        CHECK(r.contains(8));
        CHECK(!r.contains(9));
        CHECK(!r.contains(10));

        auto collected = to_vec(r);
        REQUIRE(collected.size() == 5);
        CHECK(collected[0] == 0);
        CHECK(collected[1] == 2);
        CHECK(collected[2] == 4);
        CHECK(collected[3] == 6);
        CHECK(collected[4] == 8);
    }

    SECTION("stride 3 over [1, 12)") {
        StridedIndexSet<index_type, index_type, index_type> r{1, 12, 3};
        CHECK(r.size() == 4);  // 1, 4, 7, 10
        CHECK(r.contains(1));
        CHECK(!r.contains(2));
        CHECK(!r.contains(3));
        CHECK(r.contains(4));
        CHECK(r.contains(7));
        CHECK(r.contains(10));
        CHECK(!r.contains(11));

        auto collected = to_vec(r);
        REQUIRE(collected.size() == 4);
        CHECK(collected[0] == 1);
        CHECK(collected[1] == 4);
        CHECK(collected[2] == 7);
        CHECK(collected[3] == 10);
    }

    SECTION("stride larger than range") {
        StridedIndexSet<index_type, index_type, index_type> r{0, 3, 5};
        CHECK(r.size() == 1);  // only index 0
        CHECK(r.contains(0));
        CHECK(!r.contains(3));

        auto collected = to_vec(r);
        REQUIRE(collected.size() == 1);
        CHECK(collected[0] == 0);
    }

    SECTION("empty range") {
        StridedIndexSet<index_type, index_type, index_type> r{5, 5, 2};
        CHECK(r.empty());
        CHECK(r.size() == 0);
        auto collected = to_vec(r);
        CHECK(collected.empty());
    }

    SECTION("stride overshoot past last") {
        // range(0, 7, 3) → {0, 3, 6}
        StridedIndexSet<index_type, index_type, index_type> r{0, 7, 3};
        CHECK(r.size() == 3);
        CHECK(!r.contains(7));  // 6 + 3 = 9 > 7
        auto collected = to_vec(r);
        REQUIRE(collected.size() == 3);
        CHECK(collected[2] == 6);
    }
}

TEST_CASE("StridedIndexSet: compile-time stride",
          "[index_set][strided_index_set]") {
    SECTION("stride = static 2") {
        using S2 = StridedIndexSet<index_type, index_type,
                                    static_index_t<2>>;
        S2 r{0, 10, {}};
        CHECK(r.size() == 5);
        CHECK(r.contains(0));
        CHECK(!r.contains(1));
        CHECK(r.contains(4));

        auto collected = to_vec(r);
        REQUIRE(collected.size() == 5);
        CHECK(collected[0] == 0);
        CHECK(collected[4] == 8);
    }

    SECTION("fully compile-time: first=0, last=12, stride=3") {
        using CT = StridedIndexSet<static_index_t<0>,
                                    static_index_t<12>,
                                    static_index_t<3>>;
        CT r{};
        CHECK(r.size() == 4);  // 0, 3, 6, 9
        CHECK(r.contains(0));
        CHECK(r.contains(3));
        CHECK(r.contains(6));
        CHECK(r.contains(9));
        CHECK(!r.contains(1));
        CHECK(!r.contains(11));

        auto collected = to_vec(r);
        REQUIRE(collected.size() == 4);
        CHECK(collected[0] == 0);
        CHECK(collected[3] == 9);
    }
}

TEST_CASE("StridedIndexSet: CTAD",
          "[index_set][strided_index_set][ctad]") {
    SECTION("three integral args → runtime StridedIndexSet") {
        auto r = StridedIndexSet{0, 10, 3};
        STATIC_CHECK(IsStridedIndexSet<decltype(r)>);
        // All fields should be runtime index_type
        STATIC_CHECK(std::is_same_v<decltype(r.first), index_type>);
        STATIC_CHECK(std::is_same_v<decltype(r.last), index_type>);
        STATIC_CHECK(std::is_same_v<decltype(r.stride), index_type>);
        CHECK(r.size() == 4);  // 0, 3, 6, 9
    }

    SECTION("two integral args → contiguous (stride = static 1)") {
        auto r = StridedIndexSet{2, 7};
        STATIC_CHECK(IsContiguousIndexSet<decltype(r)>);
        STATIC_CHECK(std::is_same_v<decltype(r.stride), static_index_t<1>>);
        CHECK(r.size() == 5);
        CHECK(r.contains(2));
        CHECK(r.contains(6));
        CHECK(!r.contains(7));
    }

    SECTION("integral_constant args preserved") {
        auto r = StridedIndexSet{static_index_t<0>{}, static_index_t<12>{},
                                  static_index_t<3>{}};
        STATIC_CHECK(std::is_same_v<decltype(r.first), static_index_t<0>>);
        STATIC_CHECK(std::is_same_v<decltype(r.last), static_index_t<12>>);
        STATIC_CHECK(std::is_same_v<decltype(r.stride), static_index_t<3>>);
        CHECK(r.size() == 4);
    }

    SECTION("mixed: runtime first/last, compile-time stride") {
        auto r = StridedIndexSet{0, 10, static_index_t<2>{}};
        STATIC_CHECK(std::is_same_v<decltype(r.first), index_type>);
        STATIC_CHECK(std::is_same_v<decltype(r.last), index_type>);
        STATIC_CHECK(std::is_same_v<decltype(r.stride), static_index_t<2>>);
        CHECK(r.size() == 5);
    }
}

TEST_CASE("StridedIndexSet: contiguous alias backward compat",
          "[index_set][strided_index_set]") {
    // ContiguousIndexRange is ContiguousIndexSet<> = StridedIndexSet<idx, idx, static_index_t<1>>
    ContiguousIndexRange cr{2, 5};
    CHECK(cr.size() == 3);
    CHECK(cr.contains(2));
    CHECK(cr.contains(4));
    CHECK(!cr.contains(5));

    // Should be able to access .stride field
    STATIC_CHECK(decltype(cr.stride)::value == 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 1 tests: StaticSparseIndexSet
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("StaticSparseIndexSet: basic operations",
          "[index_set][static_sparse]") {
    SECTION("3 elements") {
        StaticSparseIndexSet<3> r{{{1, 4, 7}}};
        CHECK(!r.empty());
        CHECK(r.size() == 3);
        CHECK(!r.contains(0));
        CHECK(r.contains(1));
        CHECK(!r.contains(2));
        CHECK(!r.contains(3));
        CHECK(r.contains(4));
        CHECK(!r.contains(5));
        CHECK(!r.contains(6));
        CHECK(r.contains(7));
        CHECK(!r.contains(8));

        std::vector<index_type> collected(r.begin(), r.end());
        REQUIRE(collected.size() == 3);
        CHECK(collected[0] == 1);
        CHECK(collected[1] == 4);
        CHECK(collected[2] == 7);
    }

    SECTION("1 element") {
        StaticSparseIndexSet<1> r{{{5}}};
        CHECK(!r.empty());
        CHECK(r.size() == 1);
        CHECK(r.contains(5));
        CHECK(!r.contains(0));
    }

    SECTION("empty (N=0)") {
        StaticSparseIndexSet<0> r{};
        CHECK(r.empty());
        CHECK(r.size() == 0);
        CHECK(!r.contains(0));
    }

    SECTION("linear scan for N<=8") {
        StaticSparseIndexSet<8> r{{{0, 1, 2, 3, 10, 20, 30, 40}}};
        CHECK(r.contains(0));
        CHECK(r.contains(3));
        CHECK(!r.contains(4));
        CHECK(r.contains(10));
        CHECK(r.contains(40));
        CHECK(!r.contains(41));
    }

    SECTION("binary search for N>8") {
        StaticSparseIndexSet<9> r{{{1, 3, 5, 7, 9, 11, 13, 15, 17}}};
        CHECK(r.contains(1));
        CHECK(r.contains(9));
        CHECK(r.contains(17));
        CHECK(!r.contains(0));
        CHECK(!r.contains(2));
        CHECK(!r.contains(18));
    }
}

TEST_CASE("StaticSparseIndexSet: IndexSet concept",
          "[index_set][static_sparse]") {
    STATIC_CHECK(IndexSet<StaticSparseIndexSet<0>>);
    STATIC_CHECK(IndexSet<StaticSparseIndexSet<1>>);
    STATIC_CHECK(IndexSet<StaticSparseIndexSet<5>>);
    STATIC_CHECK(IndexSet<StaticSparseIndexSet<10>>);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 1 tests: SpanSparseIndexSet
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("SpanSparseIndexSet: non-owning view",
          "[index_set][span_sparse]") {
    const std::vector<index_type> data = {2, 5, 8, 11};
    SpanSparseIndexSet r{std::span<const index_type>(data)};

    CHECK(!r.empty());
    CHECK(r.size() == 4);
    CHECK(!r.contains(0));
    CHECK(!r.contains(1));
    CHECK(r.contains(2));
    CHECK(!r.contains(3));
    CHECK(r.contains(5));
    CHECK(r.contains(8));
    CHECK(r.contains(11));
    CHECK(!r.contains(12));

    std::vector<index_type> collected(r.begin(), r.end());
    REQUIRE(collected.size() == 4);
    CHECK(collected[0] == 2);
    CHECK(collected[1] == 5);
    CHECK(collected[2] == 8);
    CHECK(collected[3] == 11);
}

TEST_CASE("SpanSparseIndexSet: empty span",
          "[index_set][span_sparse]") {
    SpanSparseIndexSet r{std::span<const index_type>{}};
    CHECK(r.empty());
    CHECK(r.size() == 0);
    CHECK(!r.contains(0));
}

TEST_CASE("SpanSparseIndexSet: to_owned() creates DynamicSparseIndexSet",
          "[index_set][span_sparse]") {
    const std::vector<index_type> data = {3, 6, 9};
    SpanSparseIndexSet span_r{std::span<const index_type>(data)};

    auto owned = span_r.to_owned();
    STATIC_CHECK(IsDynamicSparseIndexSet<decltype(owned)>);

    // owned should have same contents
    CHECK(owned.size() == 3);
    CHECK(owned.contains(3));
    CHECK(owned.contains(6));
    CHECK(owned.contains(9));
    CHECK(!owned.contains(0));

    // owned should be independent of original data
    REQUIRE(owned.indices.size() == 3);
    CHECK(owned.indices[0] == 3);
    CHECK(owned.indices[1] == 6);
    CHECK(owned.indices[2] == 9);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 1 tests: DynamicSparseIndexSet
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("DynamicSparseIndexSet: owning vector",
          "[index_set][dynamic_sparse]") {
    DynamicSparseIndexSet r{{1, 3, 7, 10}};
    CHECK(!r.empty());
    CHECK(r.size() == 4);
    CHECK(!r.contains(0));
    CHECK(r.contains(1));
    CHECK(!r.contains(2));
    CHECK(r.contains(3));
    CHECK(r.contains(7));
    CHECK(r.contains(10));
    CHECK(!r.contains(11));

    std::vector<index_type> collected(r.begin(), r.end());
    REQUIRE(collected.size() == 4);
    CHECK(collected[0] == 1);
    CHECK(collected[1] == 3);
    CHECK(collected[2] == 7);
    CHECK(collected[3] == 10);
}

TEST_CASE("DynamicSparseIndexSet: empty",
          "[index_set][dynamic_sparse]") {
    DynamicSparseIndexSet r{{}};
    CHECK(r.empty());
    CHECK(r.size() == 0);
}

TEST_CASE("DynamicSparseIndexSet: as_span() creates SpanSparseIndexSet",
          "[index_set][dynamic_sparse]") {
    DynamicSparseIndexSet owned{{2, 5, 8}};

    auto span_r = owned.as_span();
    STATIC_CHECK(IsSpanSparseIndexSet<decltype(span_r)>);

    CHECK(span_r.size() == 3);
    CHECK(span_r.contains(2));
    CHECK(span_r.contains(5));
    CHECK(span_r.contains(8));
    CHECK(!span_r.contains(0));
    CHECK(!span_r.contains(3));
}

TEST_CASE("DynamicSparseIndexSet: SparseIndexRange backward compat alias",
          "[index_set][dynamic_sparse]") {
    // SparseIndexRange is DynamicSparseIndexSet
    STATIC_CHECK(std::is_same_v<SparseIndexRange, DynamicSparseIndexSet>);
    SparseIndexRange r{{1, 3, 7, 10}};
    CHECK(r.size() == 4);
    CHECK(r.contains(3));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 1 tests: Compile-time specializations
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("ContiguousIndexSet: compile-time specialization",
          "[index_set][compile_time]") {
    using CT = ContiguousIndexSet<static_index_t<2>, static_index_t<7>>;
    CT r{};
    CHECK(!r.empty());
    CHECK(r.size() == 5);
    CHECK(!r.contains(1));
    CHECK(r.contains(2));
    CHECK(r.contains(6));
    CHECK(!r.contains(7));

    auto collected = to_vec(r);
    REQUIRE(collected.size() == 5);
    CHECK(collected[0] == 2);
    CHECK(collected[4] == 6);
}

TEST_CASE("SingleIndexSet: compile-time specialization",
          "[index_set][compile_time]") {
    using CT = SingleIndexSet<static_index_t<3>>;
    CT r{};
    CHECK(!r.empty());
    CHECK(r.size() == 1);
    CHECK(!r.contains(2));
    CHECK(r.contains(3));
    CHECK(!r.contains(4));

    auto collected = to_vec(r);
    REQUIRE(collected.size() == 1);
    CHECK(collected[0] == 3);
}

TEST_CASE("FullIndexSet: compile-time specialization",
          "[index_set][compile_time]") {
    using CT = FullIndexSet<static_index_t<5>>;
    CT r{};
    CHECK(!r.empty());
    CHECK(r.size() == 5);
    CHECK(r.contains(0));
    CHECK(r.contains(4));
    CHECK(r.contains(100));  // FullRange always returns true

    auto collected = to_vec(r);
    REQUIRE(collected.size() == 5);
    CHECK(collected[0] == 0);
    CHECK(collected[4] == 4);
}

TEST_CASE("SingleIndexSet: CTAD",
          "[index_set][compile_time][ctad]") {
    SECTION("integral → index_type") {
        auto r = SingleIndexSet{5};
        STATIC_CHECK(std::is_same_v<decltype(r.value), index_type>);
        CHECK(r.contains(5));
    }

    SECTION("integral_constant → preserved") {
        auto r = SingleIndexSet{static_index_t<7>{}};
        STATIC_CHECK(std::is_same_v<decltype(r.value), static_index_t<7>>);
        CHECK(r.contains(7));
    }
}

TEST_CASE("FullIndexSet: CTAD",
          "[index_set][compile_time][ctad]") {
    SECTION("integral → index_type") {
        auto r = FullIndexSet{10};
        STATIC_CHECK(std::is_same_v<decltype(r.extent), index_type>);
        CHECK(r.size() == 10);
    }

    SECTION("integral_constant → preserved") {
        auto r = FullIndexSet{static_index_t<8>{}};
        STATIC_CHECK(std::is_same_v<decltype(r.extent), static_index_t<8>>);
        CHECK(r.size() == 8);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 1 tests: Detection traits / concepts
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Detection traits: IsContiguousIndexSet",
          "[index_set][traits]") {
    STATIC_CHECK(IsContiguousIndexSet<ContiguousIndexRange>);
    STATIC_CHECK(IsContiguousIndexSet<ContiguousIndexSet<static_index_t<0>,
                                                           static_index_t<5>>>);
    // StridedIndexSet with non-1 stride is NOT contiguous
    STATIC_CHECK_FALSE(IsContiguousIndexSet<StridedIndexSet<index_type, index_type,
                                                         index_type>>);
    STATIC_CHECK_FALSE(IsContiguousIndexSet<StridedIndexSet<index_type, index_type,
                                                         static_index_t<2>>>);
    STATIC_CHECK_FALSE(IsContiguousIndexSet<SingleIndexRange>);
    STATIC_CHECK_FALSE(IsContiguousIndexSet<FullRange>);
}

TEST_CASE("Detection traits: IsStridedIndexSet",
          "[index_set][traits]") {
    // All StridedIndexSet specializations match, including contiguous
    STATIC_CHECK(IsStridedIndexSet<StridedIndexRange>);
    STATIC_CHECK(IsStridedIndexSet<ContiguousIndexRange>);
    STATIC_CHECK(IsStridedIndexSet<StridedIndexSet<index_type, index_type,
                                                     static_index_t<3>>>);
    STATIC_CHECK(IsStridedIndexSet<ContiguousIndexSet<static_index_t<0>,
                                                        static_index_t<5>>>);
    // Non-strided types don't match
    STATIC_CHECK_FALSE(IsStridedIndexSet<SingleIndexRange>);
    STATIC_CHECK_FALSE(IsStridedIndexSet<FullRange>);
    STATIC_CHECK_FALSE(IsStridedIndexSet<DynamicSparseIndexSet>);
}

TEST_CASE("Detection traits: IsSingleIndexSet",
          "[index_set][traits]") {
    STATIC_CHECK(IsSingleIndexSet<SingleIndexRange>);
    STATIC_CHECK(IsSingleIndexSet<SingleIndexSet<static_index_t<3>>>);
    STATIC_CHECK_FALSE(IsSingleIndexSet<ContiguousIndexRange>);
    STATIC_CHECK_FALSE(IsSingleIndexSet<FullRange>);
}

TEST_CASE("Detection traits: IsFullIndexSet",
          "[index_set][traits]") {
    STATIC_CHECK(IsFullIndexSet<FullRange>);
    STATIC_CHECK(IsFullIndexSet<FullIndexSet<static_index_t<10>>>);
    STATIC_CHECK_FALSE(IsFullIndexSet<ContiguousIndexRange>);
    STATIC_CHECK_FALSE(IsFullIndexSet<SingleIndexRange>);
}

TEST_CASE("Detection traits: IsSparseIndexSet",
          "[index_set][traits]") {
    // Individual sparse types
    STATIC_CHECK(IsStaticSparseIndexSet<StaticSparseIndexSet<3>>);
    STATIC_CHECK(IsStaticSparseIndexSet<StaticSparseIndexSet<0>>);
    STATIC_CHECK_FALSE(IsStaticSparseIndexSet<DynamicSparseIndexSet>);
    STATIC_CHECK_FALSE(IsStaticSparseIndexSet<SpanSparseIndexSet>);

    STATIC_CHECK(IsSpanSparseIndexSet<SpanSparseIndexSet>);
    STATIC_CHECK_FALSE(IsSpanSparseIndexSet<DynamicSparseIndexSet>);
    STATIC_CHECK_FALSE(IsSpanSparseIndexSet<StaticSparseIndexSet<3>>);

    STATIC_CHECK(IsDynamicSparseIndexSet<DynamicSparseIndexSet>);
    STATIC_CHECK_FALSE(IsDynamicSparseIndexSet<SpanSparseIndexSet>);
    STATIC_CHECK_FALSE(IsDynamicSparseIndexSet<StaticSparseIndexSet<3>>);

    // Union concept
    STATIC_CHECK(IsSparseIndexSet<StaticSparseIndexSet<3>>);
    STATIC_CHECK(IsSparseIndexSet<SpanSparseIndexSet>);
    STATIC_CHECK(IsSparseIndexSet<DynamicSparseIndexSet>);

    // Non-sparse types
    STATIC_CHECK_FALSE(IsSparseIndexSet<ContiguousIndexRange>);
    STATIC_CHECK_FALSE(IsSparseIndexSet<SingleIndexRange>);
    STATIC_CHECK_FALSE(IsSparseIndexSet<FullRange>);
    STATIC_CHECK_FALSE(IsSparseIndexSet<StridedIndexRange>);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 1 tests: to_contiguous_range for new types
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("to_contiguous_range: StridedIndexSet",
          "[index_set][to_contiguous_range]") {
    SECTION("stride 2 over [0, 10)") {
        StridedIndexSet r{0, 10, static_index_t<2>{}};
        // Elements: 0, 2, 4, 6, 8 → bounding box [0, 9)
        auto bb = to_contiguous_range(r);
        CHECK(bb.first == 0);
        CHECK(bb.last == 9);  // 0 + (5-1)*2 + 1 = 9
    }

    SECTION("stride 3 over [1, 12)") {
        StridedIndexSet r{1, 12, static_index_t<3>{}};
        // Elements: 1, 4, 7, 10 → bounding box [1, 11)
        auto bb = to_contiguous_range(r);
        CHECK(bb.first == 1);
        CHECK(bb.last == 11);  // 1 + (4-1)*3 + 1 = 11
    }

    SECTION("empty strided") {
        StridedIndexSet r{5, 5, static_index_t<2>{}};
        auto bb = to_contiguous_range(r);
        CHECK(bb.first == 0);
        CHECK(bb.last == 0);
    }
}

TEST_CASE("to_contiguous_range: StaticSparseIndexSet",
          "[index_set][to_contiguous_range]") {
    SECTION("normal set") {
        StaticSparseIndexSet<3> r{{{2, 5, 9}}};
        auto bb = to_contiguous_range(r);
        CHECK(bb.first == 2);
        CHECK(bb.last == 10);  // 9 + 1
    }

    SECTION("single element") {
        StaticSparseIndexSet<1> r{{{7}}};
        auto bb = to_contiguous_range(r);
        CHECK(bb.first == 7);
        CHECK(bb.last == 8);
    }

    SECTION("empty") {
        StaticSparseIndexSet<0> r{};
        auto bb = to_contiguous_range(r);
        CHECK(bb.first == 0);
        CHECK(bb.last == 0);
    }
}

TEST_CASE("to_contiguous_range: SpanSparseIndexSet",
          "[index_set][to_contiguous_range]") {
    const std::vector<index_type> data = {3, 8, 15};
    SpanSparseIndexSet r{std::span<const index_type>(data)};
    auto bb = to_contiguous_range(r);
    CHECK(bb.first == 3);
    CHECK(bb.last == 16);  // 15 + 1
}

TEST_CASE("to_contiguous_range: DynamicSparseIndexSet",
          "[index_set][to_contiguous_range]") {
    SECTION("normal set") {
        DynamicSparseIndexSet r{{1, 4, 10}};
        auto bb = to_contiguous_range(r);
        CHECK(bb.first == 1);
        CHECK(bb.last == 11);  // 10 + 1
    }

    SECTION("empty") {
        DynamicSparseIndexSet r{{}};
        auto bb = to_contiguous_range(r);
        CHECK(bb.first == 0);
        CHECK(bb.last == 0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 1 tests: Ownership round-trip
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Ownership round-trip: Dynamic → Span → Dynamic",
          "[index_set][ownership]") {
    DynamicSparseIndexSet original{{2, 5, 8, 11, 14}};

    // Dynamic → Span
    auto span_view = original.as_span();
    CHECK(span_view.size() == 5);
    CHECK(span_view.contains(2));
    CHECK(span_view.contains(14));

    // Span → Dynamic (copy)
    auto copy = span_view.to_owned();
    CHECK(copy.size() == 5);
    CHECK(copy.contains(2));
    CHECK(copy.contains(14));

    // Verify independence: modify original, copy should be unaffected
    original.indices[0] = 99;
    CHECK(!copy.contains(99));
    CHECK(copy.contains(2));

    // span_view should reflect the change (it's non-owning)
    // Note: the span still points to the same memory, but the value changed
    // The binary_search in contains() will still work correctly for the
    // new sorted order only if the vector remains sorted. After setting
    // indices[0]=99, the vector is no longer sorted so we just verify
    // the raw data is shared.
    CHECK(span_view.indices[0] == 99);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 2 tests: to_index_set() bridge functions
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("to_index_set: full_extent_t", "[index_set][to_index_set]") {
    auto r = to_index_set(full_extent_t{}, 10);
    STATIC_CHECK(std::is_same_v<decltype(r), FullRange>);
    CHECK(r.extent == 10);
    CHECK(r.size() == 10);
    CHECK(r.contains(0));
    CHECK(r.contains(9));
}

TEST_CASE("to_index_set: integer index", "[index_set][to_index_set]") {
    auto r = to_index_set(index_type{5}, 10);
    STATIC_CHECK(std::is_same_v<decltype(r), SingleIndexRange>);
    CHECK(r.value == 5);
    CHECK(r.size() == 1);
    CHECK(r.contains(5));
    CHECK(!r.contains(4));
}

TEST_CASE("to_index_set: strided_slice", "[index_set][to_index_set]") {
    SECTION("contiguous slice: offset=2, extent=5, stride=1") {
        zipper::strided_slice<index_type, index_type, index_type> s{2, 5, 1};
        auto r = to_index_set(s, 10);
        // ceil(5/1) = 5 elements: {2, 3, 4, 5, 6}
        // last = 2 + (5-1)*1 + 1 = 7
        CHECK(r.first == 2);
        CHECK(r.last == 7);
        CHECK(r.stride == 1);
        CHECK(r.size() == 5);
        auto v = to_vec(r);
        CHECK(v == std::vector<index_type>{2, 3, 4, 5, 6});
    }

    SECTION("strided slice: offset=1, extent=10, stride=3") {
        zipper::strided_slice<index_type, index_type, index_type> s{1, 10, 3};
        auto r = to_index_set(s, 20);
        // ceil(10/3) = 4 elements: {1, 4, 7, 10}
        // last = 1 + (4-1)*3 + 1 = 11
        CHECK(r.first == 1);
        CHECK(r.last == 11);
        CHECK(r.stride == 3);
        CHECK(r.size() == 4);
        auto v = to_vec(r);
        CHECK(v == std::vector<index_type>{1, 4, 7, 10});
    }

    SECTION("strided slice: offset=0, extent=6, stride=2") {
        zipper::strided_slice<index_type, index_type, index_type> s{0, 6, 2};
        auto r = to_index_set(s, 10);
        // ceil(6/2) = 3 elements: {0, 2, 4}
        // last = 0 + (3-1)*2 + 1 = 5
        CHECK(r.first == 0);
        CHECK(r.last == 5);
        CHECK(r.stride == 2);
        CHECK(r.size() == 3);
        auto v = to_vec(r);
        CHECK(v == std::vector<index_type>{0, 2, 4});
    }

    SECTION("empty slice: extent=0") {
        zipper::strided_slice<index_type, index_type, index_type> s{3, 0, 2};
        auto r = to_index_set(s, 10);
        CHECK(r.empty());
        CHECK(r.size() == 0);
    }

    SECTION("single element: extent=1, stride=5") {
        zipper::strided_slice<index_type, index_type, index_type> s{7, 1, 5};
        auto r = to_index_set(s, 20);
        // ceil(1/5) = 1 element: {7}
        // last = 7 + 0*5 + 1 = 8
        CHECK(r.first == 7);
        CHECK(r.last == 8);
        CHECK(r.size() == 1);
        auto v = to_vec(r);
        CHECK(v == std::vector<index_type>{7});
    }
}

TEST_CASE("to_index_set: strided_slice preserves stride type",
          "[index_set][to_index_set]") {
    using zipper::static_index_t;

    SECTION("static stride=1 yields ContiguousIndexSet") {
        zipper::strided_slice<index_type, index_type, static_index_t<1>> s{
            2, 5, {}};
        auto r = to_index_set(s, 10);
        // Result should be StridedIndexSet<index_type, index_type,
        // static_index_t<1>> which is ContiguousIndexSet<index_type,
        // index_type> == ContiguousIndexRange.
        STATIC_CHECK(std::is_same_v<decltype(r), ContiguousIndexRange>);
        CHECK(r.first == 2);
        CHECK(r.last == 7);
        CHECK(r.size() == 5);
    }

    SECTION("static stride=2 yields StridedIndexSet with static stride") {
        zipper::strided_slice<index_type, index_type, static_index_t<2>> s{
            1, 10, {}};
        auto r = to_index_set(s, 20);
        STATIC_CHECK(
            std::is_same_v<decltype(r),
                           StridedIndexSet<index_type, index_type,
                                           static_index_t<2>>>);
        // ceil(10/2) = 5 elements: {1, 3, 5, 7, 9}
        // last = 1 + (5-1)*2 + 1 = 10
        CHECK(r.first == 1);
        CHECK(r.last == 10);
        CHECK(r.size() == 5);
        auto v = to_vec(r);
        CHECK(v == std::vector<index_type>{1, 3, 5, 7, 9});
    }

    SECTION("runtime stride yields StridedIndexRange") {
        zipper::strided_slice<index_type, index_type, index_type> s{0, 6, 3};
        auto r = to_index_set(s, 10);
        STATIC_CHECK(std::is_same_v<decltype(r), StridedIndexRange>);
        CHECK(r.size() == 2);
    }

    SECTION("empty slice with static stride preserves type") {
        zipper::strided_slice<index_type, index_type, static_index_t<1>> s{
            3, 0, {}};
        auto r = to_index_set(s, 10);
        STATIC_CHECK(std::is_same_v<decltype(r), ContiguousIndexRange>);
        CHECK(r.empty());
        CHECK(r.size() == 0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 2 tests: range_intersection() overloads
// ═══════════════════════════════════════════════════════════════════════════════

// ── 1. FullIndexSet ∩ ... ────────────────────────────────────────────────────

TEST_CASE("range_intersection: Full ∩ Full",
          "[index_set][range_intersection]") {
    FullRange a{10};
    FullRange b{7};
    auto r = range_intersection(a, b);
    STATIC_CHECK(std::is_same_v<decltype(r), ContiguousIndexRange>);
    CHECK(r.first == 0);
    CHECK(r.last == 7);
    CHECK(r.size() == 7);
}

TEST_CASE("range_intersection: Full ∩ Contiguous",
          "[index_set][range_intersection]") {
    FullRange f{10};
    ContiguousIndexRange c{3, 7};

    SECTION("Full on left") {
        auto r = range_intersection(f, c);
        CHECK(r.first == 3);
        CHECK(r.last == 7);
    }

    SECTION("Full on right") {
        auto r = range_intersection(c, f);
        CHECK(r.first == 3);
        CHECK(r.last == 7);
    }
}

TEST_CASE("range_intersection: Full ∩ Single",
          "[index_set][range_intersection]") {
    FullRange f{10};
    SingleIndexRange s{5};

    SECTION("Full on left") {
        auto r = range_intersection(f, s);
        CHECK(r.value == 5);
    }

    SECTION("Full on right") {
        auto r = range_intersection(s, f);
        CHECK(r.value == 5);
    }
}

// ── 2. SingleIndexSet ∩ ... ──────────────────────────────────────────────────

TEST_CASE("range_intersection: Single ∩ Single",
          "[index_set][range_intersection]") {
    SECTION("same value") {
        SingleIndexRange a{5};
        SingleIndexRange b{5};
        auto r = range_intersection(a, b);
        CHECK(r.first == 5);
        CHECK(r.last == 6);
        CHECK(r.size() == 1);
    }

    SECTION("different values") {
        SingleIndexRange a{3};
        SingleIndexRange b{7};
        auto r = range_intersection(a, b);
        CHECK(r.empty());
    }
}

TEST_CASE("range_intersection: Single ∩ Contiguous",
          "[index_set][range_intersection]") {
    SingleIndexRange s{5};

    SECTION("contained") {
        ContiguousIndexRange c{3, 8};
        auto r = range_intersection(s, c);
        CHECK(r.first == 5);
        CHECK(r.last == 6);
        CHECK(r.size() == 1);
    }

    SECTION("not contained (below)") {
        ContiguousIndexRange c{6, 10};
        auto r = range_intersection(s, c);
        CHECK(r.empty());
    }

    SECTION("not contained (above)") {
        ContiguousIndexRange c{0, 5};
        auto r = range_intersection(s, c);
        CHECK(r.empty());
    }

    SECTION("symmetric") {
        ContiguousIndexRange c{3, 8};
        auto r = range_intersection(c, s);
        CHECK(r.first == 5);
        CHECK(r.last == 6);
    }
}

TEST_CASE("range_intersection: Single ∩ Strided",
          "[index_set][range_intersection]") {
    SingleIndexRange s{6};
    StridedIndexRange strided{0, 10, 3};  // {0, 3, 6, 9}

    SECTION("contained") {
        auto r = range_intersection(s, strided);
        CHECK(r.first == 6);
        CHECK(r.last == 7);
    }

    SECTION("not contained") {
        SingleIndexRange s2{7};
        auto r = range_intersection(s2, strided);
        CHECK(r.empty());
    }
}

TEST_CASE("range_intersection: Single ∩ Sparse",
          "[index_set][range_intersection]") {
    SingleIndexRange s{5};
    DynamicSparseIndexSet sparse{{2, 5, 8, 11}};

    SECTION("contained") {
        auto r = range_intersection(s, sparse);
        CHECK(r.first == 5);
        CHECK(r.last == 6);
    }

    SECTION("not contained") {
        SingleIndexRange s2{4};
        auto r = range_intersection(s2, sparse);
        CHECK(r.empty());
    }
}

// ── 3. Contiguous ∩ Contiguous ───────────────────────────────────────────────

TEST_CASE("range_intersection: Contiguous ∩ Contiguous",
          "[index_set][range_intersection]") {
    SECTION("overlapping") {
        ContiguousIndexRange a{2, 8};
        ContiguousIndexRange b{5, 12};
        auto r = range_intersection(a, b);
        CHECK(r.first == 5);
        CHECK(r.last == 8);
        CHECK(r.size() == 3);
    }

    SECTION("subset") {
        ContiguousIndexRange a{0, 10};
        ContiguousIndexRange b{3, 7};
        auto r = range_intersection(a, b);
        CHECK(r.first == 3);
        CHECK(r.last == 7);
    }

    SECTION("disjoint") {
        ContiguousIndexRange a{0, 5};
        ContiguousIndexRange b{5, 10};
        auto r = range_intersection(a, b);
        CHECK(r.empty());
    }

    SECTION("identical") {
        ContiguousIndexRange a{3, 7};
        auto r = range_intersection(a, a);
        CHECK(r.first == 3);
        CHECK(r.last == 7);
    }

    SECTION("one empty") {
        ContiguousIndexRange a{5, 5};  // empty
        ContiguousIndexRange b{3, 7};
        auto r = range_intersection(a, b);
        CHECK(r.empty());
    }

    SECTION("compile-time specialization") {
        using CT = ContiguousIndexSet<static_index_t<2>, static_index_t<8>>;
        CT a{};
        ContiguousIndexRange b{5, 12};
        auto r = range_intersection(a, b);
        CHECK(r.first == 5);
        CHECK(r.last == 8);
    }
}

// ── 4. Strided ∩ Contiguous ─────────────────────────────────────────────────

TEST_CASE("range_intersection: Strided ∩ Contiguous",
          "[index_set][range_intersection]") {
    SECTION("basic overlap") {
        // Strided: {0, 3, 6, 9, 12}  (first=0, last=13, stride=3)
        StridedIndexRange strided{0, 13, 3};
        ContiguousIndexRange contig{4, 10};
        auto r = range_intersection(strided, contig);
        // First strided element >= 4 is 6 (k=ceil(4/3)=2, 0+2*3=6)
        // new_last = min(13, 10) = 10
        CHECK(r.first == 6);
        CHECK(r.last == 10);
        CHECK(r.stride == 3);
        auto v = to_vec(r);
        CHECK(v == std::vector<index_type>{6, 9});
    }

    SECTION("contiguous starts before strided") {
        StridedIndexRange strided{5, 16, 3};  // {5, 8, 11, 14}
        ContiguousIndexRange contig{0, 20};
        auto r = range_intersection(strided, contig);
        CHECK(r.first == 5);
        CHECK(r.last == 16);
        CHECK(r.stride == 3);
        auto v = to_vec(r);
        CHECK(v == std::vector<index_type>{5, 8, 11, 14});
    }

    SECTION("contiguous window excludes all strided elements") {
        StridedIndexRange strided{0, 10, 3};  // {0, 3, 6, 9}
        ContiguousIndexRange contig{10, 20};
        auto r = range_intersection(strided, contig);
        CHECK(r.empty());
    }

    SECTION("single element intersection") {
        StridedIndexRange strided{0, 10, 5};  // {0, 5}
        ContiguousIndexRange contig{5, 6};
        auto r = range_intersection(strided, contig);
        CHECK(r.first == 5);
        CHECK(r.last == 6);
        auto v = to_vec(r);
        CHECK(v == std::vector<index_type>{5});
    }

    SECTION("symmetric: Contiguous ∩ Strided") {
        StridedIndexRange strided{0, 13, 3};  // {0, 3, 6, 9, 12}
        ContiguousIndexRange contig{4, 10};
        auto r = range_intersection(contig, strided);
        CHECK(r.first == 6);
        CHECK(r.last == 10);
        auto v = to_vec(r);
        CHECK(v == std::vector<index_type>{6, 9});
    }

    SECTION("exact alignment") {
        StridedIndexRange strided{2, 14, 4};  // {2, 6, 10}
        ContiguousIndexRange contig{2, 14};
        auto r = range_intersection(strided, contig);
        CHECK(r.first == 2);
        CHECK(r.last == 14);
        auto v = to_vec(r);
        CHECK(v == std::vector<index_type>{2, 6, 10});
    }
}

// ── 5. DisjointRange ∩ ... ──────────────────────────────────────────────────

TEST_CASE("range_intersection: DisjointRange ∩ Contiguous",
          "[index_set][range_intersection]") {
    DisjointRange<ContiguousIndexRange, ContiguousIndexRange> dr{
        std::tuple{ContiguousIndexRange{2, 5}, ContiguousIndexRange{8, 12}}};

    SECTION("intersects both segments") {
        ContiguousIndexRange c{3, 10};
        auto r = range_intersection(dr, c);
        auto v = r.indices;
        CHECK(v == std::vector<index_type>{3, 4, 8, 9});
    }

    SECTION("intersects only first segment") {
        ContiguousIndexRange c{0, 6};
        auto r = range_intersection(dr, c);
        auto v = r.indices;
        CHECK(v == std::vector<index_type>{2, 3, 4});
    }

    SECTION("intersects only second segment") {
        ContiguousIndexRange c{6, 15};
        auto r = range_intersection(dr, c);
        auto v = r.indices;
        CHECK(v == std::vector<index_type>{8, 9, 10, 11});
    }

    SECTION("intersects neither") {
        ContiguousIndexRange c{5, 8};
        auto r = range_intersection(dr, c);
        CHECK(r.empty());
    }

    SECTION("symmetric: Contiguous ∩ DisjointRange") {
        ContiguousIndexRange c{3, 10};
        auto r = range_intersection(c, dr);
        auto v = r.indices;
        CHECK(v == std::vector<index_type>{3, 4, 8, 9});
    }
}

TEST_CASE("range_intersection: DisjointRange ∩ DisjointRange",
          "[index_set][range_intersection]") {
    DisjointRange<ContiguousIndexRange, ContiguousIndexRange> a{
        std::tuple{ContiguousIndexRange{0, 5}, ContiguousIndexRange{10, 15}}};
    DisjointRange<ContiguousIndexRange, ContiguousIndexRange> b{
        std::tuple{ContiguousIndexRange{3, 8}, ContiguousIndexRange{12, 20}}};

    auto r = range_intersection(a, b);
    auto v = r.indices;
    // a[0]={0..4} ∩ b[0]={3..7} = {3,4}
    // a[0]={0..4} ∩ b[1]={12..19} = {}
    // a[1]={10..14} ∩ b[0]={3..7} = {}
    // a[1]={10..14} ∩ b[1]={12..19} = {12,13,14}
    CHECK(v == std::vector<index_type>{3, 4, 12, 13, 14});
}

// ── 6. Generic fallback ─────────────────────────────────────────────────────

TEST_CASE("range_intersection: Strided ∩ Strided (generic fallback)",
          "[index_set][range_intersection]") {
    StridedIndexRange a{0, 20, 3};  // {0, 3, 6, 9, 12, 15, 18}
    StridedIndexRange b{0, 20, 5};  // {0, 5, 10, 15}

    auto r = range_intersection(a, b);
    STATIC_CHECK(std::is_same_v<decltype(r), DynamicSparseIndexSet>);
    auto v = r.indices;
    // Common: {0, 15}
    CHECK(v == std::vector<index_type>{0, 15});
}

TEST_CASE("range_intersection: Sparse ∩ Contiguous",
          "[index_set][range_intersection]") {
    DynamicSparseIndexSet sparse{{1, 3, 5, 7, 9, 11}};
    ContiguousIndexRange contig{4, 10};

    auto r = range_intersection(sparse, contig);
    auto v = r.indices;
    CHECK(v == std::vector<index_type>{5, 7, 9});
}

TEST_CASE("range_intersection: Sparse ∩ Strided",
          "[index_set][range_intersection]") {
    DynamicSparseIndexSet sparse{{0, 2, 4, 6, 8, 10}};
    StridedIndexRange strided{0, 12, 4};  // {0, 4, 8}

    auto r = range_intersection(sparse, strided);
    auto v = r.indices;
    CHECK(v == std::vector<index_type>{0, 4, 8});
}

TEST_CASE("range_intersection: Sparse ∩ Sparse",
          "[index_set][range_intersection]") {
    DynamicSparseIndexSet a{{1, 3, 5, 7, 9}};
    DynamicSparseIndexSet b{{2, 3, 5, 8, 9}};

    auto r = range_intersection(a, b);
    auto v = r.indices;
    CHECK(v == std::vector<index_type>{3, 5, 9});
}

TEST_CASE("range_intersection: StaticSparse ∩ Contiguous",
          "[index_set][range_intersection]") {
    StaticSparseIndexSet<4> sparse{{{1, 4, 7, 10}}};
    ContiguousIndexRange contig{3, 8};

    auto r = range_intersection(sparse, contig);
    auto v = r.indices;
    CHECK(v == std::vector<index_type>{4, 7});
}

TEST_CASE("range_intersection: empty inputs",
          "[index_set][range_intersection]") {
    SECTION("empty Contiguous ∩ Contiguous") {
        ContiguousIndexRange a{5, 5};
        ContiguousIndexRange b{0, 10};
        auto r = range_intersection(a, b);
        CHECK(r.empty());
    }

    SECTION("empty Sparse ∩ anything") {
        DynamicSparseIndexSet sparse{{}};
        ContiguousIndexRange contig{0, 10};
        auto r = range_intersection(sparse, contig);
        CHECK(r.empty());
    }
}
