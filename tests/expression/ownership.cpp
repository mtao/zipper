// Tests for expression ownership semantics (refactor_all design):
// - Operator expressions on lvalue children REFERENCE their data (lazy eval)
// - Operator expressions on rvalue children OWN them (moved in)
// - View expressions (diagonal, row, col, slice, transpose, as_array) REFERENCE
// - to_owned() / eval() produce independent, owned copies
// - Expression chains with temporaries are safe
// - Assignment through mutable views

#include <zipper/Array.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/ArrayBase.hxx>
#include <zipper/expression/nullary/Constant.hpp>

#include "../catch_include.hpp"

// ============================================================
// C1: Operator expressions REFERENCE lvalue children
//     (mutations to originals are visible through lazy expression)
// ============================================================

TEST_CASE("operator_expressions_reference_lvalue_vector",
          "[ownership][operator][vector]") {
    zipper::Vector<double, 3> a{1.0, 2.0, 3.0};
    zipper::Vector<double, 3> b{10.0, 20.0, 30.0};

    auto sum = a + b;
    // Before mutation: lazy sum reads current values
    CHECK(sum(0) == 11.0);
    CHECK(sum(1) == 22.0);
    CHECK(sum(2) == 33.0);

    // Mutate originals — sum is lazy and references a, b
    a(0) = 999.0;
    b(0) = 999.0;
    CHECK(sum(0) == 999.0 + 999.0);

    // to_owned() produces an independent copy
    a(0) = 1.0;
    b(0) = 10.0;
    auto owned_sum = (a + b).to_owned();
    a(0) = 0.0;
    b(0) = 0.0;
    CHECK(owned_sum(0) == 11.0);
}

TEST_CASE("operator_expressions_reference_lvalue_difference",
          "[ownership][operator][vector]") {
    zipper::Vector<double, 3> a{10.0, 20.0, 30.0};
    zipper::Vector<double, 3> b{1.0, 2.0, 3.0};

    auto diff = a - b;
    CHECK(diff(0) == 9.0);

    a(0) = 0.0;
    b(0) = 0.0;
    CHECK(diff(0) == 0.0);  // reflects mutation
}

TEST_CASE("operator_expressions_reference_lvalue_scalar_multiply",
          "[ownership][operator][vector]") {
    zipper::Vector<double, 3> x{1.0, 2.0, 3.0};

    auto scaled = 3.0 * x;
    CHECK(scaled(0) == 3.0);

    x(0) = 999.0;
    CHECK(scaled(0) == 2997.0);  // reflects mutation
}

TEST_CASE("operator_expressions_reference_lvalue_scalar_divide",
          "[ownership][operator][vector]") {
    zipper::Vector<double, 3> x{10.0, 20.0, 30.0};

    auto divided = x / 2.0;
    CHECK(divided(0) == 5.0);

    x(0) = 999.0;
    CHECK(divided(0) == 999.0 / 2.0);  // reflects mutation
}

TEST_CASE("operator_expressions_reference_lvalue_matrix",
          "[ownership][operator][matrix]") {
    zipper::Matrix<double, 2, 2> A{{1.0, 2.0}, {3.0, 4.0}};
    zipper::Matrix<double, 2, 2> B{{10.0, 20.0}, {30.0, 40.0}};

    auto sum = A + B;
    CHECK(sum(0, 0) == 11.0);

    A(0, 0) = 999.0;
    B(0, 0) = 999.0;
    CHECK(sum(0, 0) == 999.0 + 999.0);  // reflects mutation
}

TEST_CASE("operator_expressions_reference_lvalue_array_scalar",
          "[ownership][operator][array]") {
    zipper::Array<double, 3> x;
    x(0) = 1.0;
    x(1) = 2.0;
    x(2) = 3.0;

    auto result = x + 10.0;
    CHECK(result(0) == 11.0);

    x(0) = 999.0;
    CHECK(result(0) == 1009.0);  // reflects mutation
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

    // Each row() is a view; + creates an operator expression that stores
    // references to the view objects (lvalues), which in turn reference M.
    // Since everything is lazy, mutating M IS visible.
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
// C6: Multi-level expression chains reference through intermediates
// ============================================================

TEST_CASE("multi_level_expression_chain", "[ownership][chain]") {
    zipper::Vector<double, 3> x{1.0, 2.0, 3.0};
    zipper::Vector<double, 3> y{10.0, 20.0, 30.0};

    // 2.0 * x references x (lvalue); 3.0 * y references y (lvalue)
    // The top-level + takes both as rvalues (owns them), but each
    // still holds a reference to x / y respectively.
    auto z = 2.0 * x + 3.0 * y;
    CHECK(z(0) == 2.0 + 30.0);
    CHECK(z(1) == 4.0 + 60.0);
    CHECK(z(2) == 6.0 + 90.0);

    // Mutate originals — z sees the change (references x, y)
    x(0) = 0.0;
    y(0) = 0.0;
    CHECK(z(0) == 0.0);

    // eval() breaks laziness
    x(0) = 1.0;
    y(0) = 10.0;
    auto z_eval = (2.0 * x + 3.0 * y).eval();
    x(0) = 0.0;
    CHECK(z_eval(0) == 32.0);  // unaffected by mutation
}

TEST_CASE("multi_level_matrix_expression_chain", "[ownership][chain]") {
    zipper::Matrix<double, 2, 2> A{{1.0, 2.0}, {3.0, 4.0}};
    zipper::Matrix<double, 2, 2> B{{10.0, 20.0}, {30.0, 40.0}};

    auto z = 2.0 * A + 3.0 * B;
    CHECK(z(0, 0) == 2.0 + 30.0);
    CHECK(z(0, 1) == 4.0 + 60.0);
    CHECK(z(1, 0) == 6.0 + 90.0);
    CHECK(z(1, 1) == 8.0 + 120.0);

    // Mutation visible through lazy expression
    A(0, 0) = 0.0;
    CHECK(z(0, 0) == 0.0 + 30.0);  // reflects mutation
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

    // Transpose write-through: swizzle now supports mutable views
    M.transpose()(1, 0) = 55.0;
    CHECK(M(0, 1) == 55.0);
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
// Negate operator references lvalue child
// ============================================================

TEST_CASE("negate_operator_references_lvalue", "[ownership][operator][negate]") {
    zipper::Vector<double, 3> x{1.0, 2.0, 3.0};

    auto neg = -x;
    CHECK(neg(0) == -1.0);
    CHECK(neg(1) == -2.0);
    CHECK(neg(2) == -3.0);

    // Negate stores a reference to x (lvalue) — mutation is visible
    x(0) = 999.0;
    CHECK(neg(0) == -999.0);

    // to_owned() produces an independent copy
    x(0) = 1.0;
    auto owned_neg = (-x).to_owned();
    x(0) = 0.0;
    CHECK(owned_neg(0) == -1.0);
}

// ============================================================
// to_owned() and stores_references trait
// ============================================================

// ============================================================
// unsafe() — returnable views
// ============================================================

TEST_CASE("unsafe_col_read_through", "[ownership][unsafe][col]") {
    zipper::Matrix<double, 3, 3> M{
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    // The motivating use-case: auto s = M.col(j).unsafe()
    auto c = M.col(zipper::index_type(1)).unsafe();

    // unsafe should NOT report stores_references
    STATIC_CHECK_FALSE(decltype(c)::stores_references);

    CHECK(c(0) == 2.0);
    CHECK(c(1) == 5.0);
    CHECK(c(2) == 8.0);

    // Read-through: mutations to M are visible
    M(0, 1) = 42.0;
    CHECK(c(0) == 42.0);
}

TEST_CASE("unsafe_col_write_through", "[ownership][unsafe][col]") {
    zipper::Matrix<double, 3, 3> M{
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    auto c = M.col(zipper::index_type(0)).unsafe();

    // Write-through: mutations via the view are visible in M
    c(0) = 100.0;
    c(1) = 200.0;
    c(2) = 300.0;
    CHECK(M(0, 0) == 100.0);
    CHECK(M(1, 0) == 200.0);
    CHECK(M(2, 0) == 300.0);

    // Other columns unchanged
    CHECK(M(0, 1) == 2.0);
    CHECK(M(1, 1) == 5.0);
}

TEST_CASE("unsafe_row_write_through", "[ownership][unsafe][row]") {
    zipper::Matrix<double, 2, 3> M{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    auto r = M.row(zipper::index_type(0)).unsafe();
    STATIC_CHECK_FALSE(decltype(r)::stores_references);

    r(0) = 10.0;
    r(1) = 20.0;
    r(2) = 30.0;
    CHECK(M(0, 0) == 10.0);
    CHECK(M(0, 1) == 20.0);
    CHECK(M(0, 2) == 30.0);
    // Second row unchanged
    CHECK(M(1, 0) == 4.0);
}

TEST_CASE("unsafe_const_col", "[ownership][unsafe][col]") {
    const zipper::Matrix<double, 2, 2> M{{1.0, 2.0}, {3.0, 4.0}};

    // const version — read only
    auto c = M.col(zipper::index_type(1)).unsafe();
    STATIC_CHECK_FALSE(decltype(c)::stores_references);
    CHECK(c(0) == 2.0);
    CHECK(c(1) == 4.0);
}

TEST_CASE("unsafe_diagonal_write_through",
          "[ownership][unsafe][diagonal]") {
    zipper::Matrix<double, 3, 3> M{
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    auto d = M.diagonal().unsafe();
    STATIC_CHECK_FALSE(decltype(d)::stores_references);

    d(0) = 10.0;
    d(1) = 50.0;
    d(2) = 90.0;
    CHECK(M(0, 0) == 10.0);
    CHECK(M(1, 1) == 50.0);
    CHECK(M(2, 2) == 90.0);
    // Off-diagonal unchanged
    CHECK(M(0, 1) == 2.0);
}

TEST_CASE("unsafe_vector_head_write_through",
          "[ownership][unsafe][slice]") {
    zipper::Vector<double, 4> x{1.0, 2.0, 3.0, 4.0};

    auto h = x.head<2>().unsafe();
    STATIC_CHECK_FALSE(decltype(h)::stores_references);

    h(0) = 10.0;
    h(1) = 20.0;
    CHECK(x(0) == 10.0);
    CHECK(x(1) == 20.0);
    // Tail unchanged
    CHECK(x(2) == 3.0);
    CHECK(x(3) == 4.0);
}

TEST_CASE("unsafe_assign_vector_to_col",
          "[ownership][unsafe][assign]") {
    zipper::Matrix<double, 3, 3> M{
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
    zipper::Vector<double, 3> v{100.0, 200.0, 300.0};

    auto c = M.col(zipper::index_type(2)).unsafe();
    c = v;
    CHECK(M(0, 2) == 100.0);
    CHECK(M(1, 2) == 200.0);
    CHECK(M(2, 2) == 300.0);
    // Other columns unchanged
    CHECK(M(0, 0) == 1.0);
    CHECK(M(1, 0) == 4.0);
}

TEST_CASE("unsafe_transpose_write_through",
          "[ownership][unsafe][transpose]") {
    zipper::Matrix<double, 2, 3> M{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    auto Mt = M.transpose().unsafe();
    STATIC_CHECK_FALSE(decltype(Mt)::stores_references);

    Mt(0, 1) = 99.0;
    CHECK(M(1, 0) == 99.0);
}

TEST_CASE("unsafe_lvalue_stores_reference",
          "[ownership][unsafe]") {
    zipper::Matrix<double, 2, 2> M{{1.0, 2.0}, {3.0, 4.0}};

    // Lvalue unsafe — stores a reference to M's expression
    auto uref = M.unsafe();
    STATIC_CHECK_FALSE(decltype(uref)::stores_references);

    CHECK(uref(0, 0) == 1.0);
    M(0, 0) = 42.0;
    CHECK(uref(0, 0) == 42.0);  // live view

    uref(1, 1) = 99.0;
    CHECK(M(1, 1) == 99.0);  // write-through
}

TEST_CASE("col_assign_initializer_list", "[ownership][assign][col]") {
    zipper::Matrix<double, 3, 3> M{
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    // Direct assignment to a temporary col view
    M.col(zipper::index_type(1)) = zipper::Vector<double, 3>{10.0, 20.0, 30.0};
    CHECK(M(0, 1) == 10.0);
    CHECK(M(1, 1) == 20.0);
    CHECK(M(2, 1) == 30.0);
    // Other columns unchanged
    CHECK(M(0, 0) == 1.0);
    CHECK(M(2, 2) == 9.0);
}

TEST_CASE("to_owned_produces_independent_copy",
          "[ownership][to_owned]") {
    zipper::Vector<double, 3> a{1.0, 2.0, 3.0};
    zipper::Vector<double, 3> b{10.0, 20.0, 30.0};

    auto sum = a + b;
    // sum stores references (lvalue operands)
    STATIC_CHECK(decltype(sum)::stores_references);

    auto owned = sum.to_owned();
    // owned should NOT store references
    STATIC_CHECK_FALSE(decltype(owned)::stores_references);

    a(0) = 999.0;
    // sum sees the mutation, owned does not
    CHECK(sum(0) == 999.0 + 10.0);
    CHECK(owned(0) == 11.0);
}

// ============================================================
// Slice safety: stores_references accounts for expression-typed
// index slices, and make_owned() recursively owns them.
// ============================================================

TEST_CASE("slice_with_expression_index_stores_references",
          "[ownership][slice][stores_references]") {
    zipper::Vector<double, 5> data{10.0, 20.0, 30.0, 40.0, 50.0};
    zipper::Vector<zipper::index_type, 2> idx{{1, 3}};

    // data(idx) goes through ZipperBase::operator() which calls
    // expression()(unwrapped_idx), returning a raw Slice expression.
    auto s = data(idx);
    // s is a Slice<const MDArray<double,5>&, MDArray<index_type,2>>
    // The main child is by reference → stores_references == true
    using slice_t = std::decay_t<decltype(s)>;
    STATIC_CHECK(slice_t::stores_references);

    CHECK(s(0) == 20.0);
    CHECK(s(1) == 40.0);

    // Mutation through main child reference is visible
    data(1) = 99.0;
    CHECK(s(0) == 99.0);
}

TEST_CASE("slice_make_owned_with_expression_index",
          "[ownership][slice][make_owned]") {
    zipper::Vector<double, 5> data{10.0, 20.0, 30.0, 40.0, 50.0};
    zipper::Vector<zipper::index_type, 2> idx{{1, 3}};

    auto s = data(idx);
    CHECK(s(0) == 20.0);
    CHECK(s(1) == 40.0);

    // make_owned should deep-copy both the main child AND the index slice
    auto owned = s.make_owned();
    STATIC_CHECK_FALSE(decltype(owned)::stores_references);

    CHECK(owned(0) == 20.0);
    CHECK(owned(1) == 40.0);

    // Mutations to originals should NOT affect the owned copy
    data(1) = 999.0;
    CHECK(owned(0) == 20.0);  // independent
    CHECK(s(0) == 999.0);     // live view sees mutation
}

TEST_CASE("slice_to_owned_produces_no_references",
          "[ownership][slice][to_owned]") {
    zipper::Vector<double, 5> data{1.0, 2.0, 3.0, 4.0, 5.0};
    zipper::Vector<zipper::index_type, 2> idx{{0, 2}};

    // data(idx) returns a raw Slice expression (not wrapped)
    auto s = data(idx);
    using slice_t = std::decay_t<decltype(s)>;
    STATIC_CHECK(slice_t::stores_references);

    // make_owned should produce a Slice with stores_references == false
    auto owned = s.make_owned();
    STATIC_CHECK_FALSE(std::decay_t<decltype(owned)>::stores_references);

    CHECK(owned(0) == 1.0);
    CHECK(owned(1) == 3.0);

    // Original mutation doesn't affect owned
    data(0) = 999.0;
    CHECK(owned(0) == 1.0);
}
