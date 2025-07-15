#include <iostream>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/views/nullary/ConstantView.hpp>
#include <zipper/views/nullary/IdentityView.hpp>
#include <zipper/views/nullary/RandomView.hpp>
#include <zipper/views/nullary/UnitView.hpp>
#include <zipper/views/reductions/Determinant.hpp>
#include <zipper/views/unary/PartialTraceView.hpp>
#include <zipper/views/unary/SwizzleView.hpp>

#include "../../catch_include.hpp"
#include "../../fmt_include.hpp"
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
TEST_CASE("test_determinant", "[matrix][storage][dense]") {
    {
        Matrix<double, 3, 3> A = views::nullary::IdentityView<double, 3, 3>();

        CHECK(views::reductions::Determinant(A.view())() == 1);
    }
    for (int j = 2; j < 10; ++j) {
        Matrix<double, std::dynamic_extent, std::dynamic_extent> A =
            views::nullary::IdentityView<double, std::dynamic_extent,
                                         std::dynamic_extent>(j, j);
        spdlog::info("{}", A.extent(0), A.extent(1));
        CHECK(views::reductions::Determinant(A.view())() == 1);
    }
}
