
#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
#include <iostream>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/views/nullary/ConstantView.hpp>
#include <zipper/views/nullary/IdentityView.hpp>
#include <zipper/views/nullary/RandomView.hpp>
#include <zipper/views/nullary/UnitView.hpp>
#include <zipper/views/unary/PartialTraceView.hpp>
#include <zipper/views/unary/SwizzleView.hpp>
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

struct A {
    zipper::Vector<double,3> x;
};
