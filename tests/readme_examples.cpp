// Compile-check for all code examples in README.md
// This file verifies that the examples compile (or don't compile) as documented.

#include <zipper/Vector.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Form.hpp>
#include <zipper/VectorBase.hpp>
#include <zipper/FormBase.hpp>
#include <zipper/ArrayBase.hxx>
#include <zipper/types.hpp>
#include <type_traits>

#include "catch_include.hpp"

// ============================================================
// README Section: "Principles" -- UnitView, VectorBase, FormBase
// ============================================================
// NOTE: The UnitView example uses zipper::views::nullary::unit_view which
// may be from the old API. We test what the README actually says.

// ============================================================
// README Section: "Expressions and Value Categories"
// ============================================================
TEST_CASE("readme_value_categories", "[readme]") {
    zipper::Vector<double, 4> x = {0, 1, 2, 3};
    auto p = x + 3 * zipper::Vector<double, 4>({2, 3, 4, 5});

    // p should produce correct values: x + 3*{2,3,4,5} = {0,1,2,3} + {6,9,12,15} = {6,10,14,18}
    CHECK(p(0) == 6.0);
    CHECK(p(1) == 10.0);
    CHECK(p(2) == 14.0);
    CHECK(p(3) == 18.0);
}

// ============================================================
// README Section: "Ownership" -- safe row addition from temporaries
// ============================================================
TEST_CASE("readme_safe_row_addition", "[readme]") {
    zipper::Matrix<double, 3, 3> S{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    // Safe: the + operator captures both temporary row views by value.
    auto v = S.row(std::integral_constant<zipper::index_type, 0>{})
           + S.row(std::integral_constant<zipper::index_type, 1>{});

    CHECK(v(0) == 5.0);
    CHECK(v(1) == 7.0);
    CHECK(v(2) == 9.0);
}

// ============================================================
// README Section: "The View Limitation" -- col() is NonReturnable
// ============================================================
TEST_CASE("readme_col_is_nonreturnable", "[readme]") {
    zipper::Matrix<double, 3, 3> S{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    // "auto s = S.col(...)" should compile (prvalue, no copy)
    auto s = S.col(zipper::index_type(0));
    CHECK(s(0) == 1.0);
    CHECK(s(1) == 4.0);
    CHECK(s(2) == 7.0);

    // "auto s2 = s" should NOT compile (copy deleted).
    // We verify this via type trait: s should not be copy constructible.
    using S_type = decltype(s);
    static_assert(!std::is_copy_constructible_v<S_type>,
                  "col() view should be NonReturnable (copy deleted)");
}

// ============================================================
// README Section: "Views work in local scope" -- assignment through temporary
// ============================================================
TEST_CASE("readme_assignment_through_temporary_view", "[readme]") {
    zipper::Matrix<double, 3, 3> M{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    // No unsafe() needed -- the temporary lives long enough
    M.col(zipper::index_type(1)) = zipper::Vector<double, 3>{10, 20, 30};

    CHECK(M(0, 1) == 10.0);
    CHECK(M(1, 1) == 20.0);
    CHECK(M(2, 1) == 30.0);
}

TEST_CASE("readme_mutable_row_view", "[readme]") {
    zipper::Matrix<double, 3, 3> M{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    // Live mutable view -- no copy, no allocation
    auto r = M.row(std::integral_constant<zipper::index_type, 0>{});
    r(1) = 99.0;          // writes through to M(0,1)
    CHECK(M(0, 1) == 99.0);

    M(0, 2) = 42.0;       // visible through r(2)
    CHECK(r(2) == 42.0);
}

// ============================================================
// README Section: "eval()" -- materialize to concrete type
// ============================================================
TEST_CASE("readme_eval", "[readme]") {
    zipper::Vector<double, 3> a{1, 2, 3}, b{10, 20, 30};

    auto sum_lazy = a + b;     // lazy -- still references a and b
    auto sum = (a + b).eval(); // materialized -- owns its data

    // Before mutation
    CHECK(sum_lazy(0) == 11.0);
    CHECK(sum(0) == 11.0);

    a(0) = 999.0;
    CHECK(sum_lazy(0) == 1009.0);  // sees the mutation
    CHECK(sum(0) == 11.0);         // independent copy
}

// ============================================================
// README Section: "to_owned()" -- deep copy expression tree
// ============================================================
TEST_CASE("readme_to_owned", "[readme]") {
    zipper::Vector<double, 3> a{1, 2, 3}, b{10, 20, 30};

    auto expr = 2.0 * a + 3.0 * b;     // lazy, references a and b
    auto owned = expr.to_owned();       // deep copy -- no references remain

    static_assert(!decltype(owned)::stores_references);

    // 2*{1,2,3} + 3*{10,20,30} = {2,4,6} + {30,60,90} = {32,64,96}
    CHECK(owned(0) == 32.0);

    a(0) = 0.0;
    CHECK(owned(0) == 32.0);  // fully independent
}

// ============================================================
// README Section: "unsafe()" -- copyable/returnable views
// ============================================================
TEST_CASE("readme_unsafe_copy_and_return", "[readme]") {
    zipper::Matrix<double, 3, 3> M{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    // auto s = M.col(j) works fine (prvalue, guaranteed copy elision)
    auto s = M.col(zipper::index_type(1));
    CHECK(s(0) == 2.0);

    // But copying requires unsafe()
    static_assert(!std::is_copy_constructible_v<decltype(s)>,
                  "col() view should not be copyable");
    auto s2 = s.unsafe();
    static_assert(std::is_copy_constructible_v<decltype(s2)>,
                  "unsafe() result should be copyable");
    s2(0) = 42.0;                    // writes through to M(0,1)
    CHECK(M(0, 1) == 42.0);

    // Returning from a function requires unsafe()
    auto get_col = [&M](int j) {
        return M.col(zipper::index_type(j)).unsafe();
    };
    auto c = get_col(0);
    CHECK(c(0) == 1.0);
    CHECK(c(1) == 4.0);
    CHECK(c(2) == 7.0);
}

TEST_CASE("readme_unsafe_rvalue_chaining", "[readme]") {
    zipper::Matrix<double, 3, 3> M{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    // Rvalue chaining -- Slice is moved into UnsafeRef, M's data is still live
    auto c = M.col(zipper::index_type(0)).unsafe();
    CHECK(c(0) == 1.0);
    CHECK(c(1) == 4.0);
    CHECK(c(2) == 7.0);
}

TEST_CASE("readme_unsafe_lvalue", "[readme]") {
    zipper::Matrix<double, 3, 3> M{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    // Lvalue -- UnsafeRef holds a reference to the view in `row_view`
    auto row_view = M.row(std::integral_constant<zipper::index_type, 0>{});
    auto r = row_view.unsafe();
    CHECK(r(0) == 1.0);
    CHECK(r(1) == 2.0);
    CHECK(r(2) == 3.0);
}
