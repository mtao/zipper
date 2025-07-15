

#include "../catch_include.hpp"
#include <iostream>
#include <zipper/MatrixBase.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/views/binary/CrossProductView.hpp>
#include <zipper/views/nullary/ConstantView.hpp>
#include <zipper/views/nullary/IdentityView.hpp>
#include <zipper/views/nullary/UnitView.hpp>
#include <zipper/views/unary/PartialTraceView.hpp>

TEST_CASE("test_cross_product", "[vector][products]") {
    zipper::VectorBase a = zipper::views::nullary::unit_vector<double, 3, 0>();
    zipper::VectorBase b = zipper::views::nullary::unit_vector<double>(3, 1);

    zipper::VectorBase c =
        zipper::views::binary::CrossProductView(a.view(), b.view());
    CHECK(c(0) == 0);
    CHECK(c(1) == 0);
    CHECK(c(2) == 1);
    CHECK(c == a.cross(b));

    zipper::VectorBase d =
        zipper::views::binary::CrossProductView(b.view(), a.view());
    CHECK(d(0) == 0);
    CHECK(d(1) == 0);
    CHECK(d(2) == -1);
    CHECK(d == b.cross(a));

    zipper::VectorBase e =
        zipper::views::binary::CrossProductView(a.view(), c.view());
    CHECK(e(0) == 0);
    CHECK(e(1) == -1);
    CHECK(e(2) == 0);
    CHECK(e == a.cross(c));
}

// #include <zipper/Vector.hpp>
