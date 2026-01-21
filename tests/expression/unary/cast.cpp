
#include <iostream>

#include "../../catch_include.hpp"
#include "../../fmt_include.hpp"
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/unary/Cast.hpp>
#include <zipper/utils/format.hpp>

// #include <zipper/Vector.hpp>

using namespace zipper;

TEST_CASE("test_cast", "[expression][unary]") {
  {
    using IType = expression::nullary::Identity<double, 3, 3>;
    IType i;

    auto i2 = expression::unary::cast<float>(i);
    CHECK(i2(0, 0) == 1.0f);
    CHECK(i2(1, 0) == 0.0f);
    CHECK(i2(2, 0) == 0.0f);
    CHECK(i2(0, 1) == 0.0f);
    CHECK(i2(1, 1) == 1.0f);
    CHECK(i2(2, 1) == 0.0f);
    CHECK(i2(0, 2) == 0.0f);
    CHECK(i2(1, 2) == 0.0f);
    CHECK(i2(2, 2) == 1.0f);
  }
  {
    auto A = expression::nullary::Constant<double, 3, 3>(2.0);

    auto i2 = expression::unary::cast<float>(A);
    CHECK(i2(0, 0) == 2.0f);
    CHECK(i2(1, 0) == 2.0f);
    CHECK(i2(2, 0) == 2.0f);
    CHECK(i2(0, 1) == 2.0f);
    CHECK(i2(1, 1) == 2.0f);
    CHECK(i2(2, 1) == 2.0f);
    CHECK(i2(0, 2) == 2.0f);
    CHECK(i2(1, 2) == 2.0f);
    CHECK(i2(2, 2) == 2.0f);
  }
}
