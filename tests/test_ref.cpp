/// @file test_ref.cpp
/// Tests for ref() / cref() free functions and view-propagating semantics.

#include "catch_include.hpp"
#include <type_traits>

#include <zipper/Form.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Ref.hpp>
#include <zipper/Vector.hpp>

using namespace zipper;

// ============================================================================
// Section 1: ref() / cref() basic usage
// ============================================================================

TEST_CASE("ref(vector) returns a Returnable view", "[ref][vector]") {
    Vector<double, 3> v({1.0, 2.0, 3.0});
    auto r = ref(v);

    REQUIRE(r(0) == 1.0);
    REQUIRE(r(1) == 2.0);
    REQUIRE(r(2) == 3.0);

    using R = decltype(r);
    STATIC_REQUIRE_FALSE(R::stores_references);
    STATIC_REQUIRE(std::is_copy_constructible_v<R>);
}

TEST_CASE("cref(vector) returns a const Returnable view", "[ref][vector][const]") {
    Vector<double, 3> v({1.0, 2.0, 3.0});
    auto r = cref(v);

    REQUIRE(r(0) == 1.0);
    using R = decltype(r);
    STATIC_REQUIRE_FALSE(R::stores_references);
    STATIC_REQUIRE(std::is_copy_constructible_v<R>);
}

TEST_CASE("cref(const vector) binds to const lvalue", "[ref][vector][const]") {
    const Vector<double, 3> v({1.0, 2.0, 3.0});
    auto r = cref(v);

    REQUIRE(r(0) == 1.0);
    REQUIRE(r(1) == 2.0);
    REQUIRE(r(2) == 3.0);
}

TEST_CASE("ref(matrix) works", "[ref][matrix]") {
    Matrix<double, 2, 3> m({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    auto r = ref(m);

    REQUIRE(r(0, 0) == 1.0);
    REQUIRE(r(1, 2) == 6.0);

    using R = decltype(r);
    STATIC_REQUIRE_FALSE(R::stores_references);
}

// ============================================================================
// Section 2: rvalue overloads are deleted
// ============================================================================

TEST_CASE("ref(rvalue) is deleted", "[ref][static]") {
    STATIC_REQUIRE_FALSE(
        std::is_invocable_v<decltype(static_cast<auto(*)(Vector<double,3>&) -> decltype(ref(std::declval<Vector<double,3>&>()))>(&ref<Vector<double,3>>)),
                           Vector<double, 3>&&>);
}

// ============================================================================
// Section 3: View propagation — derived views are Returnable
// ============================================================================

TEST_CASE("head() from ref(vector) is Returnable", "[ref][vector][views]") {
    Vector<double, 5> v({1.0, 2.0, 3.0, 4.0, 5.0});
    auto r = ref(v);
    auto h = r.head<3>();

    using H = decltype(h);
    STATIC_REQUIRE_FALSE(H::stores_references);
    STATIC_REQUIRE(std::is_copy_constructible_v<H>);

    REQUIRE(h(0) == 1.0);
    REQUIRE(h(1) == 2.0);
    REQUIRE(h(2) == 3.0);
}

TEST_CASE("tail() from ref(vector) is Returnable", "[ref][vector][views]") {
    Vector<double, 5> v({1.0, 2.0, 3.0, 4.0, 5.0});
    auto r = ref(v);
    auto t = r.tail<3>();

    using T = decltype(t);
    STATIC_REQUIRE_FALSE(T::stores_references);
    STATIC_REQUIRE(std::is_copy_constructible_v<T>);

    REQUIRE(t(0) == 3.0);
    REQUIRE(t(1) == 4.0);
    REQUIRE(t(2) == 5.0);
}

TEST_CASE("transpose() from ref(matrix) is Returnable", "[ref][matrix][views]") {
    Matrix<double, 2, 3> m({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    auto r = ref(m);
    auto t = r.transpose();

    using T = decltype(t);
    STATIC_REQUIRE_FALSE(T::stores_references);
    STATIC_REQUIRE(std::is_copy_constructible_v<T>);

    REQUIRE(t(0, 0) == 1.0);
    REQUIRE(t(2, 0) == 3.0);
    REQUIRE(t(0, 1) == 4.0);
}

TEST_CASE("row() from ref(matrix) is Returnable", "[ref][matrix][views]") {
    Matrix<double, 3, 3> m({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
    auto r = ref(m);
    auto row = r.row(1);

    using R = decltype(row);
    STATIC_REQUIRE_FALSE(R::stores_references);
    STATIC_REQUIRE(std::is_copy_constructible_v<R>);

    REQUIRE(row(0) == 4.0);
    REQUIRE(row(1) == 5.0);
    REQUIRE(row(2) == 6.0);
}

// ============================================================================
// Section 4: ref() reflects mutations (it's a reference)
// ============================================================================

TEST_CASE("ref(vector) reflects mutations of original", "[ref][vector][mutation]") {
    Vector<double, 3> v({1.0, 2.0, 3.0});
    auto r = ref(v);

    v(0) = 42.0;
    REQUIRE(r(0) == 42.0);
}

TEST_CASE("ref(vector) allows mutation through ref", "[ref][vector][mutation]") {
    Vector<double, 3> v({1.0, 2.0, 3.0});
    auto r = ref(v);

    r(0) = 42.0;
    REQUIRE(v(0) == 42.0);
}

TEST_CASE("head() from ref(vector) reflects mutations", "[ref][vector][views][mutation]") {
    Vector<double, 5> v({1.0, 2.0, 3.0, 4.0, 5.0});
    auto r = ref(v);
    auto h = r.head<3>();

    v(0) = 99.0;
    REQUIRE(h(0) == 99.0);
}

// ============================================================================
// Section 5: Function boundary usage
// ============================================================================

namespace {

template <concepts::Vector V>
auto get_head(const V& v) {
    return v.template head<3>();
}

template <concepts::Matrix M>
auto get_row(const M& m) {
    return m.row(1);
}

} // namespace

TEST_CASE("ref(vector) through function boundary: head", "[ref][vector][function]") {
    Vector<double, 5> v({10.0, 20.0, 30.0, 40.0, 50.0});
    auto h = get_head(ref(v));

    REQUIRE(h(0) == 10.0);
    REQUIRE(h(1) == 20.0);
    REQUIRE(h(2) == 30.0);

    // Still a live reference
    v(0) = 99.0;
    REQUIRE(h(0) == 99.0);
}

TEST_CASE("ref(matrix) through function boundary: row", "[ref][matrix][function]") {
    Matrix<double, 3, 3> m({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
    auto r = get_row(ref(m));

    REQUIRE(r(0) == 4.0);
    REQUIRE(r(1) == 5.0);
    REQUIRE(r(2) == 6.0);
}

TEST_CASE("cref(vector) through function boundary", "[ref][vector][const][function]") {
    const Vector<double, 5> v({10.0, 20.0, 30.0, 40.0, 50.0});
    auto h = get_head(cref(v));

    REQUIRE(h(0) == 10.0);
    REQUIRE(h(1) == 20.0);
    REQUIRE(h(2) == 30.0);
}

// ============================================================================
// Section 6: unsafe() does NOT propagate (existing behavior preserved)
// ============================================================================

TEST_CASE("unsafe() itself is Returnable but derived views are not",
          "[unsafe][vector][views]") {
    Vector<double, 5> v({1.0, 2.0, 3.0, 4.0, 5.0});
    auto u = v.unsafe();

    using U = decltype(u);
    STATIC_REQUIRE_FALSE(U::stores_references);
    STATIC_REQUIRE(std::is_copy_constructible_v<U>);

    // But head() from unsafe is NOT returnable (stores reference to u)
    auto h = u.head<3>();
    using H = decltype(h);
    STATIC_REQUIRE(H::stores_references);
    STATIC_REQUIRE_FALSE(std::is_copy_constructible_v<H>);
}

// ============================================================================
// Section 7: is_view_propagating trait checks
// ============================================================================

TEST_CASE("is_view_propagating trait", "[ref][traits]") {
    using V = Vector<double, 3>;
    using VExpr = V::expression_type;

    // Plain expression: not view-propagating
    using PlainTraits = expression::detail::ExpressionTraits<VExpr>;
    STATIC_REQUIRE_FALSE(PlainTraits::is_view_propagating);

    // UnsafeRef (unsafe()): not view-propagating
    using UnsafeExpr = expression::unary::UnsafeRef<VExpr &>;
    using UnsafeTraits = expression::detail::ExpressionTraits<UnsafeExpr>;
    STATIC_REQUIRE_FALSE(UnsafeTraits::is_view_propagating);

    // UnsafeRef with ViewPropagating=true (ref()): view-propagating
    using RefExpr = expression::unary::UnsafeRef<VExpr &, true>;
    using RefTraits = expression::detail::ExpressionTraits<RefExpr>;
    STATIC_REQUIRE(RefTraits::is_view_propagating);
}
