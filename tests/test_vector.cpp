#include <iostream>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/concepts/shapes.hpp>
#include <zipper/views/nullary/ConstantView.hpp>
#include <zipper/views/nullary/IdentityView.hpp>
#include <zipper/views/nullary/RandomView.hpp>
#include <zipper/views/nullary/UnitView.hpp>
#include <zipper/views/unary/PartialTraceView.hpp>
#include <zipper/views/unary/SwizzleView.hpp>

#include "catch_include.hpp"
// #include <zipper/Vector.hpp>

namespace {

void print(zipper::concepts::MatrixBaseDerived auto const& M) {
    for (zipper::index_type j = 0; j < M.extent(0); ++j) {
        for (zipper::index_type k = 0; k < M.extent(1); ++k) {
            std::cout << M(j, k) << " ";
        }
        std::cout << std::endl;
    }
}
void print(zipper::concepts::VectorBaseDerived auto const& M) {
    for (zipper::index_type j = 0; j < M.extent(0); ++j) {
        std::cout << M(j) << " ";
    }
    std::cout << std::endl;
}
}  // namespace
   //
using namespace zipper;
TEST_CASE("test_dot", "[matrix][storage][dense]") {
    Vector<double, 3> a{{0, 2, 4}};
    Vector<double, 3> b{{1, 3, 5}};

    // static_assert(zipper::concepts::ValidExtents<Vector<double,3>,3>);
    static_assert(zipper::concepts::ValidExtents<Vector<double, 3>, 3>);

    Vector c = (*a.as_form()).as_vector();

    VectorBase e0 = views::nullary::unit_vector<double, 3>(0);
    VectorBase e1 = views::nullary::unit_vector<double>(3, 1);
    VectorBase e2 = views::nullary::unit_vector<double, 3, 2>();

    CHECK(a.dot(e0) == 0);
    CHECK(a.dot(e1) == 2);
    CHECK(a.dot(e2) == 4);
    CHECK(b.dot(e0) == 1);
    CHECK(b.dot(e1) == 3);
    CHECK(b.dot(e2) == 5);
    CHECK(c.dot(e0) == 0);
    CHECK(c.dot(e1) == 2);
    CHECK(c.dot(e2) == 4);

    CHECK(a.head<2>().dot(b.head<2>()) == 6);
}
TEST_CASE("slicing", "[vector][storage][dense]") {
    Vector<double, 5> a{{0, 2, 4, 6, 8}};
    Vector<double, std::dynamic_extent> b{{1, 3, 5, 7, 9}};

    {
        auto av = a.head<2>();
        auto bv = b.head<2>();
        REQUIRE(av.extents() == create_dextents(2));
        REQUIRE(bv.extents() == create_dextents(2));
        CHECK(av(0) == 0);
        CHECK(av(1) == 2);
        CHECK(bv(0) == 1);
        CHECK(bv(1) == 3);
    }

    {
        auto av = a.head(2);
        auto bv = b.head(2);
        REQUIRE(av.extents() == create_dextents(2));
        REQUIRE(bv.extents() == create_dextents(2));
        CHECK(av(0) == 0);
        CHECK(av(1) == 2);
        CHECK(bv(0) == 1);
        CHECK(bv(1) == 3);
    }
    {
        auto av = a.tail<2>();
        auto bv = b.tail<2>();
        REQUIRE(av.extents() == create_dextents(2));
        REQUIRE(bv.extents() == create_dextents(2));
        CHECK(av(0) == 6);
        CHECK(av(1) == 8);
        CHECK(bv(0) == 7);
        CHECK(bv(1) == 9);
    }

    {
        auto av = a.tail(2);
        auto bv = b.tail(2);
        REQUIRE(av.extents() == create_dextents(2));
        REQUIRE(bv.extents() == create_dextents(2));
        CHECK(av(0) == 6);
        CHECK(av(1) == 8);
        CHECK(bv(0) == 7);
        CHECK(bv(1) == 9);
    }
    {
        auto av = a.segment<1, 2>();
        auto bv = b.segment<1, 2>();
        REQUIRE(av.extents() == create_dextents(2));
        REQUIRE(bv.extents() == create_dextents(2));
        CHECK(av(0) == 2);
        CHECK(av(1) == 4);
        CHECK(bv(0) == 3);
        CHECK(bv(1) == 5);
    }

    {
        auto av = a.segment<2>(1);
        auto bv = b.segment<2>(1);
        REQUIRE(av.extents() == create_dextents(2));
        REQUIRE(bv.extents() == create_dextents(2));
        CHECK(av(0) == 2);
        CHECK(av(1) == 4);
        CHECK(bv(0) == 3);
        CHECK(bv(1) == 5);
    }
    {
        auto av = a.segment(1, 2);
        auto bv = b.segment(1, 2);
        REQUIRE(av.extents() == create_dextents(2));
        REQUIRE(bv.extents() == create_dextents(2));
        CHECK(av(0) == 2);
        CHECK(av(1) == 4);
        CHECK(bv(0) == 3);
        CHECK(bv(1) == 5);
    }
}

struct A {
    zipper::Vector<double, 3> x;
};
