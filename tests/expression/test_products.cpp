

#include "../catch_include.hpp"
#include <iostream>
#include <zipper/MatrixBase.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/binary/CrossProduct.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/Unit.hpp>

TEST_CASE("test_cross_product", "[vector][products]") {
    zipper::VectorBase a = zipper::expression::nullary::unit_vector<double, 3, 0>();
    zipper::VectorBase b = zipper::expression::nullary::unit_vector<double>(3, 1);

    zipper::VectorBase c =
        zipper::expression::binary::CrossProduct(a.expression(), b.expression());
    CHECK(c(0) == 0);
    CHECK(c(1) == 0);
    CHECK(c(2) == 1);
    CHECK(c == a.cross(b));

    zipper::VectorBase d =
        zipper::expression::binary::CrossProduct(b.expression(), a.expression());
    CHECK(d(0) == 0);
    CHECK(d(1) == 0);
    CHECK(d(2) == -1);
    CHECK(d == b.cross(a));

    zipper::VectorBase e =
        zipper::expression::binary::CrossProduct(a.expression(), c.expression());
    CHECK(e(0) == 0);
    CHECK(e(1) == -1);
    CHECK(e(2) == 0);
    CHECK(e == a.cross(c));
}

// #include <zipper/Vector.hpp>
