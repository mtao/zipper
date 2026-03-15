// Tests for the nonzero_range infrastructure (Phase 1):
// - NonzeroRange concept satisfaction (static_asserts in header)
// - ContiguousIndexRange, SingleIndexRange, SparseIndexRange, FullRange
// - TriangularView::nonzero_range<D>() for all 6 modes
// - Identity::nonzero_range<D>() for rank-2
// - Unit::nonzero_range<0>() for rank-1
// - HasKnownZeros concept

#include <algorithm>
#include <ranges>
#include <vector>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/detail/ExpressionTraits.hpp>
#include <zipper/expression/detail/NonzeroRange.hpp>
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

TEST_CASE("ContiguousIndexRange", "[nonzero_range][range_types]") {
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

TEST_CASE("SingleIndexRange", "[nonzero_range][range_types]") {
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

TEST_CASE("SparseIndexRange", "[nonzero_range][range_types]") {
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

TEST_CASE("FullRange", "[nonzero_range][range_types]") {
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
// HasKnownZeros concept checks
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("HasKnownZeros concept", "[nonzero_range][concept]") {
    // TriangularView has known zeros
    using IdentityExpr = nullary::Identity<double, 3, 3>;
    static_assert(HasKnownZeros<IdentityExpr>);

    using UnitExpr = nullary::Unit<double, 5, index_type>;
    static_assert(HasKnownZeros<UnitExpr>);

    // MDArray (dense) does NOT have known zeros
    using DenseExpr = nullary::MDArray<double, zipper::extents<3, 3>>;
    static_assert(!HasKnownZeros<DenseExpr>);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TriangularView nonzero_range tests
// ═══════════════════════════════════════════════════════════════════════════════

static auto make_4x4() -> Matrix<double, 4, 4> {
    return Matrix<double, 4, 4>{
        {1.0, 2.0, 3.0, 4.0},
        {5.0, 6.0, 7.0, 8.0},
        {9.0, 10.0, 11.0, 12.0},
        {13.0, 14.0, 15.0, 16.0}};
}

TEST_CASE("TriangularView Lower nonzero_range",
          "[nonzero_range][triangular_view][lower]") {
    auto M = make_4x4();
    auto L = triangular_view<TriangularMode::Lower>(M);

    // nonzero_range<1>(row) = col range = [0, row+1)
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

    // nonzero_range<0>(col) = row range = [col, nrows)
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

TEST_CASE("TriangularView StrictlyLower nonzero_range",
          "[nonzero_range][triangular_view][strictly_lower]") {
    auto M = make_4x4();
    auto SL = triangular_view<TriangularMode::StrictlyLower>(M);

    // nonzero_range<1>(row) = [0, row)
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

    // nonzero_range<0>(col) = [col+1, nrows)
    SECTION("row_range_for_col") {
        auto r0 = SL.row_range_for_col(0);
        CHECK(r0.first == 1);
        CHECK(r0.last == 4);

        auto r3 = SL.row_range_for_col(3);
        CHECK(r3.empty());  // col 3 has no strictly-lower rows
    }
}

TEST_CASE("TriangularView UnitLower nonzero_range",
          "[nonzero_range][triangular_view][unit_lower]") {
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

TEST_CASE("TriangularView Upper nonzero_range",
          "[nonzero_range][triangular_view][upper]") {
    auto M = make_4x4();
    auto U = triangular_view<TriangularMode::Upper>(M);

    // nonzero_range<1>(row) = [row, ncols)
    SECTION("col_range_for_row") {
        auto r0 = U.col_range_for_row(0);
        CHECK(r0.first == 0);
        CHECK(r0.last == 4);
        CHECK(r0.size() == 4);

        auto r3 = U.col_range_for_row(3);
        CHECK(r3.first == 3);
        CHECK(r3.last == 4);
    }

    // nonzero_range<0>(col) = [0, col+1)
    SECTION("row_range_for_col") {
        auto r0 = U.row_range_for_col(0);
        CHECK(r0.first == 0);
        CHECK(r0.last == 1);

        auto r3 = U.row_range_for_col(3);
        CHECK(r3.first == 0);
        CHECK(r3.last == 4);
    }
}

TEST_CASE("TriangularView StrictlyUpper nonzero_range",
          "[nonzero_range][triangular_view][strictly_upper]") {
    auto M = make_4x4();
    auto SU = triangular_view<TriangularMode::StrictlyUpper>(M);

    // nonzero_range<1>(row) = [row+1, ncols)
    SECTION("col_range_for_row") {
        auto r0 = SU.col_range_for_row(0);
        CHECK(r0.first == 1);
        CHECK(r0.last == 4);

        auto r3 = SU.col_range_for_row(3);
        CHECK(r3.empty());
    }

    // nonzero_range<0>(col) = [0, col)
    SECTION("row_range_for_col") {
        auto r0 = SU.row_range_for_col(0);
        CHECK(r0.empty());

        auto r3 = SU.row_range_for_col(3);
        CHECK(r3.first == 0);
        CHECK(r3.last == 3);
    }
}

TEST_CASE("TriangularView UnitUpper nonzero_range",
          "[nonzero_range][triangular_view][unit_upper]") {
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

// Verify that nonzero_range results are consistent with coeff() values
TEST_CASE("TriangularView nonzero_range consistency with coeff",
          "[nonzero_range][triangular_view][consistency]") {
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

// ═══════════════════════════════════════════════════════════════════════════════
// Identity nonzero_range tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Identity nonzero_range rank-2",
          "[nonzero_range][identity]") {
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

TEST_CASE("Identity nonzero_range general rank",
          "[nonzero_range][identity]") {
    // Rank-3 Identity: nonzero at (i,i,i)
    nullary::Identity<double, 3, 3, 3> I3;

    auto r0 = I3.nonzero_range<0>(2);
    CHECK(r0.size() == 1);
    CHECK(r0.value == 2);

    auto r1 = I3.nonzero_range<1>(1);
    CHECK(r1.size() == 1);
    CHECK(r1.value == 1);

    auto r2 = I3.nonzero_range<2>(0);
    CHECK(r2.size() == 1);
    CHECK(r2.value == 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit nonzero_range tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Unit nonzero_range", "[nonzero_range][unit]") {
    SECTION("static extent, dynamic index") {
        auto u = nullary::unit_vector<double, 5>(2);
        auto r = u.nonzero_range<0>();
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
        auto r = u.nonzero_range<0>();
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

TEST_CASE("ZeroPreservingUnaryOp concept", "[nonzero_range][traits]") {
    using namespace zipper::expression::detail;

    // negate is zero-preserving
    static_assert(ZeroPreservingUnaryOp<std::negate<double>>);
    static_assert(ZeroPreservingUnaryOp<std::negate<int>>);

    // logical_not is NOT zero-preserving: !0 == true
    static_assert(!ZeroPreservingUnaryOp<std::logical_not<double>>);

    // bit_not is NOT zero-preserving: ~0 != 0
    static_assert(!ZeroPreservingUnaryOp<std::bit_not<int>>);
}

TEST_CASE("ZeroPreservingScalarOp concept", "[nonzero_range][traits]") {
    using namespace zipper::expression::detail;

    // multiplies: 0*s == 0 and s*0 == 0
    static_assert(ZeroPreservingScalarOp<std::multiplies<double>, true>);
    static_assert(ZeroPreservingScalarOp<std::multiplies<double>, false>);

    // divides: 0/s == 0 (scalar on right only)
    static_assert(ZeroPreservingScalarOp<std::divides<double>, true>);
    static_assert(!ZeroPreservingScalarOp<std::divides<double>, false>);

    // plus: 0+s != 0 in general
    static_assert(!ZeroPreservingScalarOp<std::plus<double>, true>);
    static_assert(!ZeroPreservingScalarOp<std::plus<double>, false>);

    // minus: 0-s != 0 in general
    static_assert(!ZeroPreservingScalarOp<std::minus<double>, true>);
    static_assert(!ZeroPreservingScalarOp<std::minus<double>, false>);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Transitivity: UnsafeRef
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("UnsafeRef propagates has_known_zeros",
          "[nonzero_range][transitivity][unsafe_ref]") {
    auto M = make_4x4();
    auto L = triangular_view<TriangularMode::Lower>(M);

    // UnsafeRef wrapping a TriangularView should have known zeros
    auto uref = unary::UnsafeRef(L);
    using uref_type = decltype(uref);
    using uref_traits = zipper::expression::detail::ExpressionTraits<
        std::decay_t<uref_type>>;
    static_assert(uref_traits::has_known_zeros);

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
        static_assert(!dense_traits::has_known_zeros);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Transitivity: Swizzle (Transpose)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Swizzle/Transpose propagates has_known_zeros",
          "[nonzero_range][transitivity][swizzle]") {
    auto M = make_4x4();
    auto L = triangular_view<TriangularMode::Lower>(M);

    // Transpose of Lower = Upper (structurally)
    auto LT = unary::Swizzle<decltype(L), 1, 0>(L);
    using lt_type = decltype(LT);
    using lt_traits = zipper::expression::detail::ExpressionTraits<
        std::decay_t<lt_type>>;
    static_assert(lt_traits::has_known_zeros);

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
        static_assert(!sw_traits::has_known_zeros);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Transitivity: CoefficientWiseOperation (Negate)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Negate propagates has_known_zeros",
          "[nonzero_range][transitivity][cwiseop]") {
    auto M = make_4x4();
    auto U = triangular_view<TriangularMode::Upper>(M);

    // Negate of TriangularView should preserve known zeros
    auto negU = unary::CoefficientWiseOperation<decltype(U), std::negate<double>>(U);
    using neg_type = decltype(negU);
    using neg_traits = zipper::expression::detail::ExpressionTraits<
        std::decay_t<neg_type>>;
    static_assert(neg_traits::has_known_zeros);

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
        static_assert(!lnot_traits::has_known_zeros);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Transitivity: ScalarOperation (ScalarMultiplies, ScalarDivides)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("ScalarMultiplies propagates has_known_zeros",
          "[nonzero_range][transitivity][scalarop]") {
    auto M = make_4x4();
    auto L = triangular_view<TriangularMode::Lower>(M);

    // expr * scalar (ScalarOnRight = true)
    using MulRight = unary::ScalarMultiplies<double, decltype(L), true>;
    using mul_right_traits = zipper::expression::detail::ExpressionTraits<
        std::decay_t<MulRight>>;
    static_assert(mul_right_traits::has_known_zeros);

    // scalar * expr (ScalarOnRight = false)
    using MulLeft = unary::ScalarMultiplies<double, decltype(L), false>;
    using mul_left_traits = zipper::expression::detail::ExpressionTraits<
        std::decay_t<MulLeft>>;
    static_assert(mul_left_traits::has_known_zeros);

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
          "[nonzero_range][transitivity][scalarop]") {
    auto M = make_4x4();
    auto L = triangular_view<TriangularMode::Lower>(M);

    // expr / scalar (ScalarOnRight = true): 0/s == 0
    using DivRight = unary::ScalarDivides<double, decltype(L), true>;
    using div_right_traits = zipper::expression::detail::ExpressionTraits<
        std::decay_t<DivRight>>;
    static_assert(div_right_traits::has_known_zeros);

    // scalar / expr (ScalarOnRight = false): s/0 is undefined, not zero
    using DivLeft = unary::ScalarDivides<double, decltype(L), false>;
    using div_left_traits = zipper::expression::detail::ExpressionTraits<
        std::decay_t<DivLeft>>;
    static_assert(!div_left_traits::has_known_zeros);
}

TEST_CASE("ScalarPlus does NOT propagate has_known_zeros",
          "[nonzero_range][transitivity][scalarop]") {
    auto M = make_4x4();
    auto L = triangular_view<TriangularMode::Lower>(M);

    // expr + scalar: 0 + s != 0
    using PlusRight = unary::ScalarPlus<double, decltype(L), true>;
    using plus_traits = zipper::expression::detail::ExpressionTraits<
        std::decay_t<PlusRight>>;
    static_assert(!plus_traits::has_known_zeros);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Transitivity: Slice on TriangularView
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Slice propagates has_known_zeros from TriangularView",
          "[nonzero_range][transitivity][slice]") {
    auto M = make_4x4();
    auto L = triangular_view<TriangularMode::Lower>(M);

    SECTION("full_extent slice preserves ranges exactly") {
        // Slice taking all rows, all cols = identity
        auto sl = unary::Slice(L, full_extent_t{}, full_extent_t{});
        using sl_type = decltype(sl);
        using sl_traits = zipper::expression::detail::ExpressionTraits<
            std::decay_t<sl_type>>;
        static_assert(sl_traits::has_known_zeros);

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
        // has_known_zeros is true because child has it.
        auto row_slice = unary::Slice(L, index_type{2}, full_extent_t{});
        using rs_type = decltype(row_slice);
        using rs_traits = zipper::expression::detail::ExpressionTraits<
            std::decay_t<rs_type>>;
        static_assert(rs_traits::has_known_zeros);

        // For a rank-1 slice, nonzero_range<0>() queries along
        // the only output dimension (cols). The child dimension for
        // "cols" is 1, and the fixed index is row=2.
        // L.nonzero_range<1>(2) = [0, 3), so the output should be [0, 3).
        auto r = row_slice.template nonzero_range<0>(2);
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
        static_assert(sub_traits::has_known_zeros);

        // Output row 0 maps to child row 1.
        // L.nonzero_range<1>(1) = [0, 2)
        // Remap through full_extent on cols: [0, 2)
        auto r0 = sub.col_range_for_row(0);
        CHECK(r0.first == 0);
        CHECK(r0.last == 2);

        // Output row 1 maps to child row 2.
        // L.nonzero_range<1>(2) = [0, 3)
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
        static_assert(sub_traits::has_known_zeros);

        // Output col 0 maps to child col 1.
        // U.nonzero_range<0>(1) = [0, 2) (rows with non-zeros in child col 1)
        // Remap through full_extent on rows: [0, 2)
        auto r0 = sub.row_range_for_col(0);
        CHECK(r0.first == 0);
        CHECK(r0.last == 2);

        // Output col 1 maps to child col 2.
        // U.nonzero_range<0>(2) = [0, 3)
        auto r1 = sub.row_range_for_col(1);
        CHECK(r1.first == 0);
        CHECK(r1.last == 3);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Transitivity: chained operations
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Chained transitivity: Negate(Transpose(TriangularView))",
          "[nonzero_range][transitivity][chained]") {
    auto M = make_4x4();
    auto L = triangular_view<TriangularMode::Lower>(M);

    // Transpose(Lower) -> structurally Upper
    auto LT = unary::Swizzle<decltype(L), 1, 0>(L);

    // Negate(Transpose(Lower))
    auto negLT = unary::CoefficientWiseOperation<decltype(LT), std::negate<double>>(LT);

    using neg_traits = zipper::expression::detail::ExpressionTraits<
        std::decay_t<decltype(negLT)>>;
    static_assert(neg_traits::has_known_zeros);

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
