// Tests for zero-aware addition/subtraction (Phase 3):
// - Type dispatch: dense + dense → Operation, sparse + dense → ZeroAwareOperation
// - Coefficient correctness for ZeroAwareOperation (+ and -)
// - index_set propagation (union semantics) for rank-2 and rank-1
// - All semantic wrapper types: Vector, Matrix, Form, Array, Tensor

#include <type_traits>

#include <zipper/Array.hpp>
#include <zipper/Form.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/binary/ArithmeticExpressions.hpp>
#include <zipper/expression/binary/ZeroAwareOperation.hpp>
#include <zipper/expression/detail/ExpressionTraits.hpp>
#include <zipper/expression/detail/IndexSet.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/Unit.hpp>
#include <zipper/expression/unary/TriangularView.hpp>

#include <zipper/MatrixBase.hxx>
#include <zipper/VectorBase.hxx>
#include <zipper/FormBase.hxx>
#include <zipper/ArrayBase.hxx>
#include <zipper/TensorBase.hxx>

#include "../catch_include.hpp"

using namespace zipper;
using namespace zipper::expression;
using namespace zipper::expression::detail;

// ═══════════════════════════════════════════════════════════════════════════════
// Helper: check whether an expression type is a ZeroAwareOperation
// ═══════════════════════════════════════════════════════════════════════════════

template <typename T>
constexpr bool is_zero_aware_operation_v = false;

template <typename A, typename B, typename Op>
constexpr bool is_zero_aware_operation_v<
    binary::ZeroAwareOperation<A, B, Op>> = true;

// Check if a wrapper's underlying expression is ZeroAwareOperation.
template <typename WrapperExpr>
constexpr bool wrapper_has_zero_aware_expr() {
    using expr_t = typename std::decay_t<WrapperExpr>::expression_type;
    return is_zero_aware_operation_v<std::decay_t<expr_t>>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Type dispatch tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Zero-aware dispatch: dense + dense stays regular Operation",
          "[zero_aware][dispatch]") {
    Matrix<double, 3, 3> A;
    A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    Matrix<double, 3, 3> B;
    B = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};

    auto result = A + B;
    using result_expr_t =
        std::decay_t<decltype(result.expression())>;

    // Both A and B are dense → should NOT produce ZeroAwareOperation
    CHECK_FALSE(is_zero_aware_operation_v<result_expr_t>);
    CHECK_FALSE(HasIndexSet<result_expr_t>);
}

TEST_CASE("Zero-aware dispatch: identity + dense → ZeroAwareOperation",
          "[zero_aware][dispatch]") {
    Matrix<double, 3, 3> A;
    A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    MatrixBase I = nullary::Identity<double, 3, 3>{};

    auto result = I + A;
    using result_expr_t =
        std::decay_t<decltype(result.expression())>;

    // Identity has known zeros → result should be ZeroAwareOperation
    CHECK(is_zero_aware_operation_v<result_expr_t>);
    CHECK(HasIndexSet<result_expr_t>);
}

TEST_CASE("Zero-aware dispatch: triangular + triangular → ZeroAwareOperation",
          "[zero_aware][dispatch]") {
    Matrix<double, 3, 3> A;
    A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    auto L = MatrixBase(triangular_view<TriangularMode::Lower>(A));
    auto U = MatrixBase(triangular_view<TriangularMode::Upper>(A));

    auto result = L + U;
    using result_expr_t =
        std::decay_t<decltype(result.expression())>;

    CHECK(is_zero_aware_operation_v<result_expr_t>);
    CHECK(HasIndexSet<result_expr_t>);
}

TEST_CASE("Zero-aware dispatch: subtraction also dispatches",
          "[zero_aware][dispatch]") {
    Matrix<double, 3, 3> A;
    A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    MatrixBase I = nullary::Identity<double, 3, 3>{};

    auto result = A - I;
    using result_expr_t =
        std::decay_t<decltype(result.expression())>;

    CHECK(is_zero_aware_operation_v<result_expr_t>);
    CHECK(HasIndexSet<result_expr_t>);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Coefficient correctness tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Zero-aware addition: coefficients are correct",
          "[zero_aware][coefficients]") {
    Matrix<double, 3, 3> A;
    A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    MatrixBase I = nullary::Identity<double, 3, 3>{};

    SECTION("I + A") {
        auto result = I + A;
        // I + A should be A with diagonal += 1
        CHECK(result(0, 0) == Catch::Approx(2.0));
        CHECK(result(0, 1) == Catch::Approx(2.0));
        CHECK(result(0, 2) == Catch::Approx(3.0));
        CHECK(result(1, 0) == Catch::Approx(4.0));
        CHECK(result(1, 1) == Catch::Approx(6.0));
        CHECK(result(1, 2) == Catch::Approx(6.0));
        CHECK(result(2, 0) == Catch::Approx(7.0));
        CHECK(result(2, 1) == Catch::Approx(8.0));
        CHECK(result(2, 2) == Catch::Approx(10.0));
    }

    SECTION("A + I") {
        auto result = A + I;
        CHECK(result(0, 0) == Catch::Approx(2.0));
        CHECK(result(1, 1) == Catch::Approx(6.0));
        CHECK(result(2, 2) == Catch::Approx(10.0));
        CHECK(result(0, 1) == Catch::Approx(2.0));
        CHECK(result(1, 0) == Catch::Approx(4.0));
    }
}

TEST_CASE("Zero-aware subtraction: coefficients are correct",
          "[zero_aware][coefficients]") {
    Matrix<double, 3, 3> A;
    A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    MatrixBase I = nullary::Identity<double, 3, 3>{};

    SECTION("A - I") {
        auto result = A - I;
        CHECK(result(0, 0) == Catch::Approx(0.0));
        CHECK(result(0, 1) == Catch::Approx(2.0));
        CHECK(result(1, 1) == Catch::Approx(4.0));
        CHECK(result(2, 2) == Catch::Approx(8.0));
    }

    SECTION("I - A") {
        auto result = I - A;
        CHECK(result(0, 0) == Catch::Approx(0.0));
        CHECK(result(0, 1) == Catch::Approx(-2.0));
        CHECK(result(1, 0) == Catch::Approx(-4.0));
        CHECK(result(1, 1) == Catch::Approx(-4.0));
    }
}

TEST_CASE("Zero-aware addition: triangular views",
          "[zero_aware][coefficients][triangular]") {
    Matrix<double, 3, 3> A;
    A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    auto L = MatrixBase(triangular_view<TriangularMode::Lower>(A));
    auto U = MatrixBase(triangular_view<TriangularMode::Upper>(A));

    auto result = L + U;

    // L + U should give: original A on the diagonal, L's values below,
    // U's values above. On the diagonal, both contribute:
    // L(i,i) + U(i,i) = A(i,i) + A(i,i) = 2*A(i,i)
    CHECK(result(0, 0) == Catch::Approx(2.0));  // 1+1
    CHECK(result(1, 1) == Catch::Approx(10.0)); // 5+5
    CHECK(result(2, 2) == Catch::Approx(18.0)); // 9+9

    CHECK(result(1, 0) == Catch::Approx(4.0));  // L has 4, U has 0
    CHECK(result(2, 0) == Catch::Approx(7.0));  // L has 7, U has 0
    CHECK(result(2, 1) == Catch::Approx(8.0));  // L has 8, U has 0

    CHECK(result(0, 1) == Catch::Approx(2.0));  // L has 0, U has 2
    CHECK(result(0, 2) == Catch::Approx(3.0));  // L has 0, U has 3
    CHECK(result(1, 2) == Catch::Approx(6.0));  // L has 0, U has 6
}

// ═══════════════════════════════════════════════════════════════════════════════
// Index set propagation tests (union semantics)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Zero-aware: index_set is union of children (Lower + Upper)",
          "[zero_aware][index_set]") {
    Matrix<double, 4, 4> A;
    A = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

    auto L = MatrixBase(triangular_view<TriangularMode::Lower>(A));
    auto U = MatrixBase(triangular_view<TriangularMode::Upper>(A));

    auto sum = L + U;
    auto& expr = sum.expression();

    // L col_range_for_row(0) = [0, 1), U col_range_for_row(0) = [0, 4)
    // union = [0, 4)
    SECTION("col range for row 0") {
        auto r = to_contiguous_range(expr.col_range_for_row(0));
        CHECK(r.first == 0);
        CHECK(r.last == 4);
    }

    // L col_range_for_row(3) = [0, 4), U col_range_for_row(3) = [3, 4)
    // union = [0, 4)
    SECTION("col range for row 3") {
        auto r = to_contiguous_range(expr.col_range_for_row(3));
        CHECK(r.first == 0);
        CHECK(r.last == 4);
    }

    // L row_range_for_col(0) = [0, 4), U row_range_for_col(0) = [0, 1)
    // union = [0, 4)
    SECTION("row range for col 0") {
        auto r = to_contiguous_range(expr.row_range_for_col(0));
        CHECK(r.first == 0);
        CHECK(r.last == 4);
    }
}

TEST_CASE("Zero-aware: index_set with identity",
          "[zero_aware][index_set]") {
    Matrix<double, 4, 4> A;
    A = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

    MatrixBase I = nullary::Identity<double, 4, 4>{};

    auto sum = I + A;
    auto& expr = sum.expression();

    // Identity col_range_for_row(r) = SingleIndexRange{r} → [r, r+1)
    // A is dense → FullRange [0, 4)
    // union = [0, 4) (A dominates)
    SECTION("col range for row 0") {
        auto r = to_contiguous_range(expr.col_range_for_row(0));
        CHECK(r.first == 0);
        CHECK(r.last == 4);
    }

    SECTION("col range for row 2") {
        auto r = to_contiguous_range(expr.col_range_for_row(2));
        CHECK(r.first == 0);
        CHECK(r.last == 4);
    }
}

TEST_CASE("Zero-aware: index_set strictly-triangular sum",
          "[zero_aware][index_set]") {
    Matrix<double, 4, 4> A;
    A = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

    auto SL = MatrixBase(triangular_view<TriangularMode::StrictlyLower>(A));
    auto SU = MatrixBase(triangular_view<TriangularMode::StrictlyUpper>(A));

    auto sum = SL + SU;
    auto& expr = sum.expression();

    // SL col_range_for_row(0) = [0, 0) (empty)
    // SU col_range_for_row(0) = [1, 4)
    // union = [1, 4)
    SECTION("col range for row 0") {
        auto r = to_contiguous_range(expr.col_range_for_row(0));
        CHECK(r.first == 1);
        CHECK(r.last == 4);
    }

    // SL col_range_for_row(1) = [0, 1)
    // SU col_range_for_row(1) = [2, 4)
    // union = [0, 4)
    SECTION("col range for row 1") {
        auto r = to_contiguous_range(expr.col_range_for_row(1));
        CHECK(r.first == 0);
        CHECK(r.last == 4);
    }

    // SL col_range_for_row(3) = [0, 3)
    // SU col_range_for_row(3) = [4, 4) (empty)
    // union = [0, 3)
    SECTION("col range for row 3") {
        auto r = to_contiguous_range(expr.col_range_for_row(3));
        CHECK(r.first == 0);
        CHECK(r.last == 3);
    }

    // Row ranges
    // SL row_range_for_col(0) = [1, 4)
    // SU row_range_for_col(0) = [0, 0) (empty)
    // union = [1, 4)
    SECTION("row range for col 0") {
        auto r = to_contiguous_range(expr.row_range_for_col(0));
        CHECK(r.first == 1);
        CHECK(r.last == 4);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Vector tests (rank-1)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Zero-aware dispatch: unit vector + dense vector",
          "[zero_aware][vector][dispatch]") {
    Vector<double, 4> v;
    v = {1.0, 2.0, 3.0, 4.0};

    auto e2 = VectorBase(nullary::Unit<double, 4, index_type>(2));

    auto result = e2 + v;
    using result_expr_t =
        std::decay_t<decltype(result.expression())>;

    // Unit has known zeros
    CHECK(is_zero_aware_operation_v<result_expr_t>);
    CHECK(HasIndexSet<result_expr_t>);

    // Coefficients
    CHECK(result(0) == Catch::Approx(1.0));
    CHECK(result(1) == Catch::Approx(2.0));
    CHECK(result(2) == Catch::Approx(4.0));  // 1 + 3
    CHECK(result(3) == Catch::Approx(4.0));
}

TEST_CASE("Zero-aware: unit + unit nonzero_segment",
          "[zero_aware][vector][index_set]") {
    auto e1 = VectorBase(nullary::Unit<double, 6, index_type>(1));
    auto e4 = VectorBase(nullary::Unit<double, 6, index_type>(4));

    auto sum = e1 + e4;
    auto& expr = sum.expression();

    // e1 nonzero_segment = [1, 2), e4 = [4, 5)
    // union = [1, 5)
    auto r = to_contiguous_range(expr.nonzero_segment());
    CHECK(r.first == 1);
    CHECK(r.last == 5);

    // Coefficient check
    CHECK(sum(0) == Catch::Approx(0.0));
    CHECK(sum(1) == Catch::Approx(1.0));
    CHECK(sum(2) == Catch::Approx(0.0));
    CHECK(sum(3) == Catch::Approx(0.0));
    CHECK(sum(4) == Catch::Approx(1.0));
    CHECK(sum(5) == Catch::Approx(0.0));
}

TEST_CASE("Zero-aware: unit - unit",
          "[zero_aware][vector][subtraction]") {
    auto e1 = VectorBase(nullary::Unit<double, 4, index_type>(1));
    auto e3 = VectorBase(nullary::Unit<double, 4, index_type>(3));

    auto diff = e1 - e3;
    CHECK(diff(0) == Catch::Approx(0.0));
    CHECK(diff(1) == Catch::Approx(1.0));
    CHECK(diff(2) == Catch::Approx(0.0));
    CHECK(diff(3) == Catch::Approx(-1.0));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Assignment to owning type
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Zero-aware result can be assigned to owning Matrix",
          "[zero_aware][assignment]") {
    Matrix<double, 3, 3> A;
    A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    MatrixBase I = nullary::Identity<double, 3, 3>{};

    Matrix<double, 3, 3> result;
    result = I + A;

    CHECK(result(0, 0) == Catch::Approx(2.0));
    CHECK(result(0, 1) == Catch::Approx(2.0));
    CHECK(result(1, 0) == Catch::Approx(4.0));
    CHECK(result(1, 1) == Catch::Approx(6.0));
    CHECK(result(2, 2) == Catch::Approx(10.0));
}

TEST_CASE("Zero-aware result can be assigned to owning Vector",
          "[zero_aware][assignment]") {
    Vector<double, 4> v;
    v = {10.0, 20.0, 30.0, 40.0};

    auto e2 = VectorBase(nullary::Unit<double, 4, index_type>(2));

    Vector<double, 4> result;
    result = v + e2;

    CHECK(result(0) == Catch::Approx(10.0));
    CHECK(result(1) == Catch::Approx(20.0));
    CHECK(result(2) == Catch::Approx(31.0));
    CHECK(result(3) == Catch::Approx(40.0));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Chaining: zero-aware + zero-aware should still be zero-aware
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Zero-aware chaining: (I + A) + L is still zero-aware",
          "[zero_aware][chaining]") {
    Matrix<double, 3, 3> A;
    A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    MatrixBase I = nullary::Identity<double, 3, 3>{};
    auto L = MatrixBase(triangular_view<TriangularMode::Lower>(A));

    // I + A → ZeroAwareOperation (has_index_set = true)
    auto step1 = I + A;
    // (I + A) + L → both have known zeros → ZeroAwareOperation
    auto step2 = step1 + L;
    using step2_expr_t =
        std::decay_t<decltype(step2.expression())>;

    CHECK(is_zero_aware_operation_v<step2_expr_t>);
    CHECK(HasIndexSet<step2_expr_t>);

    // Value check: I + A + L at (0,0) = 1 + 1 + 1 = 3
    CHECK(step2(0, 0) == Catch::Approx(3.0));
    // at (0, 2) = 0 + 3 + 0 = 3
    CHECK(step2(0, 2) == Catch::Approx(3.0));
    // at (2, 0) = 0 + 7 + 7 = 14
    CHECK(step2(2, 0) == Catch::Approx(14.0));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Form tests
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Zero-aware: form subtraction with unit",
          "[zero_aware][form]") {
    Form<double, 4> f;
    f(0) = 1.0; f(1) = 2.0; f(2) = 3.0; f(3) = 4.0;

    // Make a unit form: we need a FormBase wrapping a Unit
    auto e0 = FormBase(nullary::Unit<double, 4, index_type>(0));

    auto result = f - e0;
    using result_expr_t =
        std::decay_t<decltype(result.expression())>;

    CHECK(is_zero_aware_operation_v<result_expr_t>);

    CHECK(result(0) == Catch::Approx(0.0));  // 1 - 1
    CHECK(result(1) == Catch::Approx(2.0));
    CHECK(result(2) == Catch::Approx(3.0));
    CHECK(result(3) == Catch::Approx(4.0));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dense + dense stays non-zero-aware for vectors too
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Dense vector + dense vector stays regular Operation",
          "[zero_aware][vector][dispatch]") {
    Vector<double, 3> a, b;
    a = {1.0, 2.0, 3.0};
    b = {4.0, 5.0, 6.0};

    auto result = a + b;
    using result_expr_t =
        std::decay_t<decltype(result.expression())>;

    CHECK_FALSE(is_zero_aware_operation_v<result_expr_t>);
    CHECK_FALSE(HasIndexSet<result_expr_t>);

    CHECK(result(0) == Catch::Approx(5.0));
    CHECK(result(1) == Catch::Approx(7.0));
    CHECK(result(2) == Catch::Approx(9.0));
}

// ═══════════════════════════════════════════════════════════════════════════════
// ZeroAware + DisjointRange tests (OffDiagonal operand)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Zero-aware dispatch: OffDiagonal + dense → ZeroAwareOperation",
          "[zero_aware][dispatch][off_diagonal]") {
    Matrix<double, 3, 3> A;
    A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    auto OD = MatrixBase(triangular_view<TriangularMode::OffDiagonal>(A));

    Matrix<double, 3, 3> B;
    B = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};

    auto result = OD + B;
    using result_expr_t = std::decay_t<decltype(result.expression())>;

    CHECK(is_zero_aware_operation_v<result_expr_t>);
    CHECK(HasIndexSet<result_expr_t>);
}

TEST_CASE("Zero-aware: OffDiagonal + dense coefficients correct",
          "[zero_aware][coefficients][off_diagonal]") {
    Matrix<double, 3, 3> A;
    A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    auto OD = MatrixBase(triangular_view<TriangularMode::OffDiagonal>(A));

    Matrix<double, 3, 3> B;
    B = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};

    auto result = OD + B;

    // OD has zero on diagonal, child values elsewhere
    // result = OD + B = (A with diag zeroed) + B
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double od_val = (i == j) ? 0.0 : A(i, j);
            double expected = od_val + B(i, j);
            CHECK(result(i, j) == Catch::Approx(expected));
        }
    }
}

TEST_CASE("Zero-aware: OffDiagonal + Identity index_set is union",
          "[zero_aware][index_set][off_diagonal]") {
    Matrix<double, 4, 4> A;
    A = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

    auto OD = MatrixBase(triangular_view<TriangularMode::OffDiagonal>(A));
    MatrixBase I = nullary::Identity<double, 4, 4>{};

    auto sum = OD + I;
    auto& expr = sum.expression();

    // OffDiagonal col_range_for_row(r) = [0,r) ∪ [r+1,4) (DisjointRange)
    // Identity col_range_for_row(r) = SingleIndexRange{r} → [r, r+1)
    // Union should cover [0, 4) for all rows
    for (index_type row = 0; row < 4; ++row) {
        auto r = to_contiguous_range(expr.col_range_for_row(row));
        CHECK(r.first == 0);
        CHECK(r.last == 4);
    }

    // Coefficient check: OD(i,j) + I(i,j)
    for (index_type i = 0; i < 4; ++i) {
        for (index_type j = 0; j < 4; ++j) {
            double od_val = (i == j) ? 0.0 : A(i, j);
            double id_val = (i == j) ? 1.0 : 0.0;
            CHECK(sum(i, j) == Catch::Approx(od_val + id_val));
        }
    }
}

TEST_CASE("Zero-aware: OffDiagonal + OffDiagonal",
          "[zero_aware][index_set][off_diagonal]") {
    Matrix<double, 3, 3> A;
    A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    Matrix<double, 3, 3> B;
    B = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};

    auto OD_A = MatrixBase(triangular_view<TriangularMode::OffDiagonal>(A));
    auto OD_B = MatrixBase(triangular_view<TriangularMode::OffDiagonal>(B));

    auto sum = OD_A + OD_B;
    auto& expr = sum.expression();

    CHECK(is_zero_aware_operation_v<std::decay_t<decltype(expr)>>);

    // Both have DisjointRange col ranges; union should still be disjoint
    // (same pattern for both)
    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double a_val = (i == j) ? 0.0 : A(i, j);
            double b_val = (i == j) ? 0.0 : B(i, j);
            CHECK(sum(i, j) == Catch::Approx(a_val + b_val));
        }
    }

    // On diagonal, both are zero → sum is zero
    for (index_type i = 0; i < 3; ++i) {
        CHECK(sum(i, i) == Catch::Approx(0.0));
    }
}

TEST_CASE("Zero-aware: OffDiagonal - Lower coefficients correct",
          "[zero_aware][coefficients][off_diagonal]") {
    Matrix<double, 3, 3> A;
    A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    auto OD = MatrixBase(triangular_view<TriangularMode::OffDiagonal>(A));
    auto L  = MatrixBase(triangular_view<TriangularMode::Lower>(A));

    auto diff = OD - L;

    for (index_type i = 0; i < 3; ++i) {
        for (index_type j = 0; j < 3; ++j) {
            double od_val = (i == j) ? 0.0 : A(i, j);
            double l_val = (j <= i) ? A(i, j) : 0.0;
            CHECK(diff(i, j) == Catch::Approx(od_val - l_val));
        }
    }
}
