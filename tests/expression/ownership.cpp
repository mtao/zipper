// Tests for expression ownership semantics:
// - Operator expressions (+ - * /) OWN copies of their children
// - View expressions (diagonal, row, col, slice, transpose, as_array) REFERENCE
// - Expression chains with temporaries are safe
// - eval() materializes and breaks laziness
// - Assignment through mutable views

#include <zipper/Array.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/ArrayBase.hxx>
#include <zipper/expression/nullary/Constant.hpp>

#include "../catch_include.hpp"

// ============================================================
// C1: Operator expressions own copies (independence from originals)
// ============================================================

TEST_CASE("operator_expressions_own_copies_vector",
          "[ownership][operator][vector]") {
    zipper::Vector<double, 3> a{1.0, 2.0, 3.0};
    zipper::Vector<double, 3> b{10.0, 20.0, 30.0};

    auto sum = a + b;
    // Mutate originals after creating the expression
    a(0) = 999.0;
    b(0) = 999.0;

    // sum should be unaffected — it owns copies
    CHECK(sum(0) == 11.0);
    CHECK(sum(1) == 22.0);
    CHECK(sum(2) == 33.0);
}

TEST_CASE("operator_expressions_own_copies_difference",
          "[ownership][operator][vector]") {
    zipper::Vector<double, 3> a{10.0, 20.0, 30.0};
    zipper::Vector<double, 3> b{1.0, 2.0, 3.0};

    auto diff = a - b;
    a(0) = 0.0;
    b(0) = 0.0;

    CHECK(diff(0) == 9.0);
    CHECK(diff(1) == 18.0);
    CHECK(diff(2) == 27.0);
}

TEST_CASE("operator_expressions_own_copies_scalar_multiply",
          "[ownership][operator][vector]") {
    zipper::Vector<double, 3> x{1.0, 2.0, 3.0};

    auto scaled = 3.0 * x;
    x(0) = 999.0;

    CHECK(scaled(0) == 3.0);
    CHECK(scaled(1) == 6.0);
    CHECK(scaled(2) == 9.0);
}

TEST_CASE("operator_expressions_own_copies_scalar_divide",
          "[ownership][operator][vector]") {
    zipper::Vector<double, 3> x{10.0, 20.0, 30.0};

    auto divided = x / 2.0;
    x(0) = 999.0;

    CHECK(divided(0) == 5.0);
    CHECK(divided(1) == 10.0);
    CHECK(divided(2) == 15.0);
}

TEST_CASE("operator_expressions_own_copies_matrix",
          "[ownership][operator][matrix]") {
    zipper::Matrix<double, 2, 2> A{{1.0, 2.0}, {3.0, 4.0}};
    zipper::Matrix<double, 2, 2> B{{10.0, 20.0}, {30.0, 40.0}};

    auto sum = A + B;
    A(0, 0) = 999.0;
    B(0, 0) = 999.0;

    CHECK(sum(0, 0) == 11.0);
    CHECK(sum(0, 1) == 22.0);
    CHECK(sum(1, 0) == 33.0);
    CHECK(sum(1, 1) == 44.0);
}

TEST_CASE("operator_expressions_own_copies_array_scalar",
          "[ownership][operator][array]") {
    zipper::Array<double, 3> x;
    x(0) = 1.0;
    x(1) = 2.0;
    x(2) = 3.0;

    auto result = x + 10.0;
    x(0) = 999.0;

    CHECK(result(0) == 11.0);
    CHECK(result(1) == 12.0);
    CHECK(result(2) == 13.0);
}

// ============================================================
// C2: View expressions reference (reflect mutations)
// ============================================================

TEST_CASE("view_diagonal_reflects_mutations",
          "[ownership][view][diagonal]") {
    zipper::Matrix<double, 3, 3> M{
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    auto d = M.diagonal();
    CHECK(d(0) == 1.0);
    CHECK(d(1) == 5.0);
    CHECK(d(2) == 9.0);

    // Mutate the matrix
    M(1, 1) = 42.0;
    // View should reflect the change
    CHECK(d(1) == 42.0);
}

TEST_CASE("view_row_reflects_mutations", "[ownership][view][row]") {
    zipper::Matrix<double, 3, 3> M{
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    auto r = M.row(std::integral_constant<zipper::index_type, 1>{});
    CHECK(r(0) == 4.0);
    CHECK(r(1) == 5.0);
    CHECK(r(2) == 6.0);

    M(1, 2) = 99.0;
    CHECK(r(2) == 99.0);
}

TEST_CASE("view_col_reflects_mutations", "[ownership][view][col]") {
    zipper::Matrix<double, 3, 3> M{
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    auto c = M.col(std::integral_constant<zipper::index_type, 0>{});
    CHECK(c(0) == 1.0);
    CHECK(c(1) == 4.0);
    CHECK(c(2) == 7.0);

    M(2, 0) = 77.0;
    CHECK(c(2) == 77.0);
}

TEST_CASE("view_transpose_reflects_mutations",
          "[ownership][view][transpose]") {
    zipper::Matrix<double, 2, 3> M{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    auto Mt = M.transpose();
    CHECK(Mt(0, 0) == 1.0);
    CHECK(Mt(1, 0) == 2.0);
    CHECK(Mt(0, 1) == 4.0);

    M(0, 1) = 99.0;
    CHECK(Mt(1, 0) == 99.0);
}

TEST_CASE("view_topRows_reflects_mutations",
          "[ownership][view][slice]") {
    zipper::Matrix<double, 3, 3> M{
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    auto top = M.topRows<2>();
    CHECK(top(0, 0) == 1.0);
    CHECK(top(1, 2) == 6.0);

    M(0, 0) = 42.0;
    CHECK(top(0, 0) == 42.0);
}

TEST_CASE("view_head_reflects_mutations", "[ownership][view][slice]") {
    zipper::Vector<double, 4> x{1.0, 2.0, 3.0, 4.0};

    auto h = x.head<3>();
    CHECK(h(0) == 1.0);
    CHECK(h(1) == 2.0);
    CHECK(h(2) == 3.0);

    x(1) = 77.0;
    CHECK(h(1) == 77.0);
}

TEST_CASE("view_tail_reflects_mutations", "[ownership][view][slice]") {
    zipper::Vector<double, 4> x{1.0, 2.0, 3.0, 4.0};

    auto t = x.tail<2>();
    CHECK(t(0) == 3.0);
    CHECK(t(1) == 4.0);

    x(3) = 55.0;
    CHECK(t(1) == 55.0);
}

// ============================================================
// C3: Expression chains with temporaries are safe
// ============================================================

TEST_CASE("chain_as_array_plus_scalar", "[ownership][chain]") {
    zipper::Vector<double, 3> x{1.0, 2.0, 3.0};

    // as_array() creates a view on lvalue x; + 1 creates owning expression
    auto xa = x.as_array();
    auto y = xa + 1.0;
    CHECK(y(0) == 2.0);
    CHECK(y(1) == 3.0);
    CHECK(y(2) == 4.0);
}

TEST_CASE("chain_row_plus_row", "[ownership][chain]") {
    zipper::Matrix<double, 2, 3> M{{1.0, 2.0, 3.0}, {10.0, 20.0, 30.0}};

    // Each row() is a view; + creates an operator expression that owns copies
    // of the view objects, but those views still reference M's data.
    // Since the binary operation is lazy (evaluates on access), mutating M
    // DOES affect the result.
    auto r0 = M.row(std::integral_constant<zipper::index_type, 0>{});
    auto r1 = M.row(std::integral_constant<zipper::index_type, 1>{});
    auto sum = r0 + r1;
    CHECK(sum(0) == 11.0);
    CHECK(sum(1) == 22.0);
    CHECK(sum(2) == 33.0);

    // Mutate M — sum IS affected because the views reference M's storage
    M(0, 0) = 999.0;
    CHECK(sum(0) == 999.0 + 10.0);

    // To get an independent copy, use eval()
    M(0, 0) = 1.0; // restore
    auto sum_eval = (r0 + r1).eval();
    M(0, 0) = 999.0;
    CHECK(sum_eval(0) == 11.0); // unaffected
}

// ============================================================
// C6: Multi-level expression chains
// ============================================================

TEST_CASE("multi_level_expression_chain", "[ownership][chain]") {
    zipper::Vector<double, 3> x{1.0, 2.0, 3.0};
    zipper::Vector<double, 3> y{10.0, 20.0, 30.0};

    auto z = 2.0 * x + 3.0 * y;
    CHECK(z(0) == 2.0 + 30.0);
    CHECK(z(1) == 4.0 + 60.0);
    CHECK(z(2) == 6.0 + 90.0);

    // Mutate originals — z should be unaffected
    x(0) = 0.0;
    y(0) = 0.0;
    CHECK(z(0) == 32.0);
}

TEST_CASE("multi_level_matrix_expression_chain", "[ownership][chain]") {
    zipper::Matrix<double, 2, 2> A{{1.0, 2.0}, {3.0, 4.0}};
    zipper::Matrix<double, 2, 2> B{{10.0, 20.0}, {30.0, 40.0}};

    auto z = 2.0 * A + 3.0 * B;
    CHECK(z(0, 0) == 2.0 + 30.0);
    CHECK(z(0, 1) == 4.0 + 60.0);
    CHECK(z(1, 0) == 6.0 + 90.0);
    CHECK(z(1, 1) == 8.0 + 120.0);

    A(0, 0) = 0.0;
    CHECK(z(0, 0) == 32.0);
}

// ============================================================
// C7: eval() materializes expression trees
// ============================================================

TEST_CASE("eval_materializes_vector", "[ownership][eval]") {
    zipper::Vector<double, 3> x{1.0, 2.0, 3.0};
    zipper::Vector<double, 3> y{10.0, 20.0, 30.0};

    auto v = (x + y).eval();
    // v is a materialized Vector — independent of x and y
    y(0) = 999.0;
    CHECK(v(0) == 11.0);
    CHECK(v(1) == 22.0);
    CHECK(v(2) == 33.0);
}

TEST_CASE("eval_materializes_matrix", "[ownership][eval]") {
    zipper::Matrix<double, 2, 2> A{{1.0, 2.0}, {3.0, 4.0}};

    auto v = (2.0 * A).eval();
    A(0, 0) = 999.0;
    CHECK(v(0, 0) == 2.0);
    CHECK(v(0, 1) == 4.0);
    CHECK(v(1, 0) == 6.0);
    CHECK(v(1, 1) == 8.0);
}

TEST_CASE("eval_breaks_view_laziness", "[ownership][eval]") {
    zipper::Matrix<double, 3, 3> M{
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    // diagonal() is a lazy view, but eval() should materialize it
    auto d = M.diagonal().eval();
    M(0, 0) = 999.0;
    CHECK(d(0) == 1.0);  // eval'd copy is independent
    CHECK(d(1) == 5.0);
    CHECK(d(2) == 9.0);
}

// ============================================================
// C8: Assignment through mutable views
// ============================================================

TEST_CASE("assignment_through_row_view", "[ownership][assign][view]") {
    zipper::Matrix<double, 3, 3> M{
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
    zipper::Vector<double, 3> v{100.0, 200.0, 300.0};

    M.row(std::integral_constant<zipper::index_type, 0>{}) = v;
    CHECK(M(0, 0) == 100.0);
    CHECK(M(0, 1) == 200.0);
    CHECK(M(0, 2) == 300.0);
    // Other rows unchanged
    CHECK(M(1, 0) == 4.0);
    CHECK(M(2, 0) == 7.0);
}

TEST_CASE("assignment_through_col_view", "[ownership][assign][view]") {
    zipper::Matrix<double, 3, 3> M{
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
    zipper::Vector<double, 3> v{100.0, 200.0, 300.0};

    M.col(std::integral_constant<zipper::index_type, 1>{}) = v;
    CHECK(M(0, 1) == 100.0);
    CHECK(M(1, 1) == 200.0);
    CHECK(M(2, 1) == 300.0);
    // Other cols unchanged
    CHECK(M(0, 0) == 1.0);
    CHECK(M(1, 0) == 4.0);
}

TEST_CASE("assignment_through_diagonal_view",
          "[ownership][assign][view]") {
    zipper::Matrix<double, 3, 3> M{
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    M.diagonal() =
        zipper::expression::nullary::Constant<double, 3>(0.0);
    CHECK(M(0, 0) == 0.0);
    CHECK(M(1, 1) == 0.0);
    CHECK(M(2, 2) == 0.0);
    // Off-diagonal unchanged
    CHECK(M(0, 1) == 2.0);
    CHECK(M(1, 0) == 4.0);
}

TEST_CASE("assignment_through_head_view", "[ownership][assign][view]") {
    zipper::Vector<double, 4> x{1.0, 2.0, 3.0, 4.0};

    x.head<2>() = zipper::expression::nullary::Constant<double, 2>(0.0);
    CHECK(x(0) == 0.0);
    CHECK(x(1) == 0.0);
    CHECK(x(2) == 3.0);
    CHECK(x(3) == 4.0);
}

TEST_CASE("element_write_through_view", "[ownership][assign][view]") {
    zipper::Matrix<double, 2, 2> M{{1.0, 2.0}, {3.0, 4.0}};

    auto r = M.row(std::integral_constant<zipper::index_type, 0>{});
    r(0) = 99.0;
    CHECK(M(0, 0) == 99.0);

    auto d = M.diagonal();
    d(1) = 77.0;
    CHECK(M(1, 1) == 77.0);

    // Note: transpose() returns a const view (swizzle is const-only),
    // so write-through transpose is not supported.
    // M.transpose()(1, 0) = 55.0; // would not compile
}

// ============================================================
// Compound: construct from expression evaluates eagerly
// ============================================================

TEST_CASE("vector_construction_from_expression_evaluates",
          "[ownership][eval]") {
    zipper::Vector<double, 3> x{1.0, 2.0, 3.0};
    zipper::Vector<double, 3> y{10.0, 20.0, 30.0};

    // Constructing a Vector from an expression tree should materialize
    zipper::Vector v = x + y;
    x(0) = 999.0;
    CHECK(v(0) == 11.0);
    CHECK(v(1) == 22.0);
    CHECK(v(2) == 33.0);
}

TEST_CASE("matrix_construction_from_expression_evaluates",
          "[ownership][eval]") {
    zipper::Matrix<double, 2, 2> A{{1.0, 2.0}, {3.0, 4.0}};
    zipper::Matrix<double, 2, 2> B{{10.0, 20.0}, {30.0, 40.0}};

    zipper::Matrix C = A + B;
    A(0, 0) = 999.0;
    CHECK(C(0, 0) == 11.0);
    CHECK(C(0, 1) == 22.0);
}

// ============================================================
// Compound-assign operators
// ============================================================

TEST_CASE("compound_assign_operators_vector",
          "[ownership][assign][operator]") {
    zipper::Vector<double, 3> x{1.0, 2.0, 3.0};
    zipper::Vector<double, 3> y{10.0, 20.0, 30.0};

    x += y;
    CHECK(x(0) == 11.0);
    CHECK(x(1) == 22.0);
    CHECK(x(2) == 33.0);

    x -= y;
    CHECK(x(0) == 1.0);
    CHECK(x(1) == 2.0);
    CHECK(x(2) == 3.0);

    x *= 2.0;
    CHECK(x(0) == 2.0);
    CHECK(x(1) == 4.0);
    CHECK(x(2) == 6.0);

    x /= 2.0;
    CHECK(x(0) == 1.0);
    CHECK(x(1) == 2.0);
    CHECK(x(2) == 3.0);
}

// ============================================================
// Negate operator ownership
// ============================================================

TEST_CASE("negate_operator_owns_copy", "[ownership][operator][negate]") {
    zipper::Vector<double, 3> x{1.0, 2.0, 3.0};

    auto neg = -x;
    x(0) = 999.0;

    CHECK(neg(0) == -1.0);
    CHECK(neg(1) == -2.0);
    CHECK(neg(2) == -3.0);
}
