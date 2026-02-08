#include <iostream>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/reductions/Determinant.hpp>

#include "../../catch_include.hpp"
#include "../../fmt_include.hpp"

namespace {

void print(zipper::concepts::Matrix auto const& M) {
    for (zipper::index_type j = 0; j < M.extent(0); ++j) {
        for (zipper::index_type k = 0; k < M.extent(1); ++k) {
            std::cout << M(j, k) << " ";
        }
        std::cout << std::endl;
    }
}
void print(zipper::concepts::Vector auto const& M) {
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
        Matrix<double, 3, 3> A = expression::nullary::Identity<double, 3, 3>();

        CHECK(expression::reductions::Determinant(A.expression())() == 1);
    }
    for (int j = 2; j < 10; ++j) {
        Matrix<double, std::dynamic_extent, std::dynamic_extent> A =
            expression::nullary::Identity<double, std::dynamic_extent,
                                         std::dynamic_extent>(j, j);
        //fmt::print("{} {}\n", A.extent(0), A.extent(1));
        CHECK(expression::reductions::Determinant(A.expression())() == 1);
    }
}
