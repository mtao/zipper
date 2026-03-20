#include <print>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/reductions/Determinant.hpp>

#include "../../catch_include.hpp"

namespace {

void print(zipper::concepts::Matrix auto const& M) {
    for (zipper::index_type j = 0; j < M.extent(0); ++j) {
        for (zipper::index_type k = 0; k < M.extent(1); ++k) {
            std::print("{} ", M(j, k));
        }
        std::println("");
    }
}
void print(zipper::concepts::Vector auto const& M) {
    for (zipper::index_type j = 0; j < M.extent(0); ++j) {
        std::print("{} ", M(j));
    }
    std::println("");
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
        CHECK(expression::reductions::Determinant(A.expression())() == 1);
    }
}
