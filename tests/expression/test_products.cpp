

#include "../catch_include.hpp"
#include <iostream>
#include <zipper/MatrixBase.hxx>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/binary/CrossProduct.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/Unit.hpp>

TEST_CASE("test_cross_product", "[vector][products]") {
    zipper::VectorBase a = zipper::expression::nullary::unit_vector<double, 3, 0>();
    zipper::VectorBase b = zipper::expression::nullary::unit_vector<double>(3, 1);

    // Use the wrapper API cross() which constructs via std::in_place
    // (expressions that store references are NonReturnable by design)
    auto c = a.cross(b);
    CHECK(c(0) == 0);
    CHECK(c(1) == 0);
    CHECK(c(2) == 1);

    auto d = b.cross(a);
    CHECK(d(0) == 0);
    CHECK(d(1) == 0);
    CHECK(d(2) == -1);

    auto e = a.cross(c);
    CHECK(e(0) == 0);
    CHECK(e(1) == -1);
    CHECK(e(2) == 0);
}

// #include <zipper/Vector.hpp>
