

#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
#include <uvl/Matrix.hpp>

TEST_CASE("test_all_extents", "[storage][dense]") {
    uvl::Matrix<double, 4, 4> L;
    uvl::Matrix<double, 4, std::dynamic_extent> M(4);

    auto X = L * M;

    uvl::Matrix<double, 4, 4> C(X + M);
}
