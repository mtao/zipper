#include "catch_include.hpp"
#include <iostream>
#include <zipper/Vector.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/expression/unary/UnsafeRef.hpp>
using namespace zipper;
TEST_CASE("test_expression", "[vector][storage][dense][span]") {

    Vector<int,5> v({4,5,3,5,6});

    auto div_3 = v.unary_expr([](int x) -> bool { return x % 3 == 0; });

    REQUIRE(div_3.extent(0) == 5);
    CHECK_FALSE(div_3(0));
    CHECK_FALSE(div_3(1));
    CHECK(div_3(2));
    CHECK_FALSE(div_3(3));
    CHECK(div_3(4));
}

TEST_CASE("unsafe_ref stores_references flag", "[unsafe_ref]") {
    // UnsafeRef should force stores_references to false
    using MDArr = expression::nullary::MDArray<double, extents<3>>;
    using RefChild = const MDArr &;
    using UR = expression::unary::UnsafeRef<RefChild>;

    // The child reference type normally has stores_references == true
    static_assert(
        expression::unary::detail::DefaultUnaryExpressionTraits<RefChild>::stores_references == true);

    // UnsafeRef overrides it to false
    static_assert(
        expression::detail::ExpressionTraits<UR>::stores_references == false);
}

TEST_CASE("unsafe_ref value pass-through", "[unsafe_ref]") {
    Vector<double, 3> v({1.0, 2.0, 3.0});
    auto ur = v.unsafe_ref();

    // stores_references should be false on the wrapper
    STATIC_REQUIRE(!std::decay_t<decltype(ur)>::stores_references);

    REQUIRE(ur(0) == 1.0);
    REQUIRE(ur(1) == 2.0);
    REQUIRE(ur(2) == 3.0);
}

TEST_CASE("unsafe_ref is copyable", "[unsafe_ref]") {
    Vector<double, 3> v({10.0, 20.0, 30.0});
    auto ur = v.unsafe_ref();

    // Copy should work (NonReturnable would delete copy ctor)
    auto ur2 = ur;
    REQUIRE(ur2(0) == 10.0);
    REQUIRE(ur2(1) == 20.0);
    REQUIRE(ur2(2) == 30.0);
}

TEST_CASE("unsafe_ref reflects mutations", "[unsafe_ref]") {
    Vector<double, 3> v({1.0, 2.0, 3.0});
    auto ur = v.unsafe_ref();

    // Mutate the original
    v(0) = 42.0;

    // unsafe_ref sees the change (it's a reference)
    REQUIRE(ur(0) == 42.0);
}

TEST_CASE("unsafe_ref can be returned from function", "[unsafe_ref]") {
    Vector<double, 3> v({5.0, 6.0, 7.0});

    // Simulate returning from a function
    auto make_ref = [&v]() {
        return v.unsafe_ref();
    };
    auto ur = make_ref();
    REQUIRE(ur(0) == 5.0);
    REQUIRE(ur(1) == 6.0);
    REQUIRE(ur(2) == 7.0);
}

TEST_CASE("unsafe_ref make_owned delegates to child", "[unsafe_ref]") {
    Vector<double, 3> v({1.0, 2.0, 3.0});
    auto ur = v.unsafe_ref();

    // to_owned() on the wrapper should produce a fully-owned copy
    auto owned = ur.to_owned();
    STATIC_REQUIRE(!std::decay_t<decltype(owned)>::stores_references);

    REQUIRE(owned(0) == 1.0);
    REQUIRE(owned(1) == 2.0);
    REQUIRE(owned(2) == 3.0);

    // Mutating original should NOT affect the owned copy
    v(0) = 999.0;
    REQUIRE(owned(0) == 1.0);
}
