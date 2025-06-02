#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
#include <iostream>
#include <zipper/MatrixBase.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/views/nullary/ConstantView.hpp>
#include <zipper/views/nullary/IdentityView.hpp>
#include <zipper/views/nullary/RandomView.hpp>
#include <zipper/views/nullary/UnitView.hpp>
#include <zipper/views/unary/HomogeneousView.hpp>
#include <zipper/views/unary/PartialTraceView.hpp>

TEST_CASE("test_homogeneous", "[vector][homogeneous]") {
    zipper::Vector a = zipper::views::nullary::unit_vector<double, 3, 0>();
    zipper::Vector b = zipper::views::nullary::unit_vector<double>(3, 1);

    auto c = zipper::views::unary::HomogeneousView(a.view());
    CHECK(c(0) == a(0));
    CHECK(c(1) == a(1));
    CHECK(c(2) == a(2));
    CHECK(c(3) == 1);
    CHECK(zipper::VectorBase(c) == a.homogeneous());
    auto d = zipper::views::unary::HomogeneousView(b.view());
    CHECK(d(0) == b(0));
    CHECK(d(1) == b(1));
    CHECK(d(2) == b(2));
    CHECK(d(3) == 1);
    CHECK(zipper::VectorBase(d) == b.homogeneous());
}

// #include <zipper/Vector.hpp>
