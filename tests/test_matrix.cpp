

#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
#include <iostream>
#include <uvl/Matrix.hpp>
#include <uvl/Vector.hpp>
#include <uvl/views/nullary/ConstantView.hpp>
#include <uvl/views/nullary/IdentityView.hpp>
#include <uvl/views/nullary/RandomView.hpp>
#include <uvl/views/unary/SwizzleView.hpp>
// #include <uvl/Vector.hpp>

namespace {

void print(uvl::concepts::MatrixBaseDerived auto const& M) {
    for (uvl::index_type j = 0; j < M.extent(0); ++j) {
        for (uvl::index_type k = 0; k < M.extent(1); ++k) {
            std::cout << M(j, k) << " ";
        }
        std::cout << std::endl;
    }
}
void print(uvl::concepts::VectorBaseDerived auto const& M) {
    for (uvl::index_type j = 0; j < M.extent(0); ++j) {
        std::cout << M(j) << " ";
    }
    std::cout << std::endl;
}
}  // namespace
TEST_CASE("test_all_extents", "[storage][dense]") {
    uvl::Matrix<double, 3, 3> I =
        uvl::MatrixBase(uvl::views::nullary::IdentityView<double, 3, 3>{});
    uvl::Matrix<double, 3, std::dynamic_extent> M(3);
    uvl::Vector<double, 3> x;

    x(0) = 2;
    x(1) = 5;
    x(2) = 9;

    M(0, 0) = 0;
    M(1, 0) = 1;
    M(2, 0) = 2;
    M(0, 1) = 3;
    M(1, 1) = 4;
    M(2, 1) = 5;
    M(0, 2) = 6;
    M(1, 2) = 7;
    M(2, 2) = 8;

    auto MC = M.cast<double>();
    auto X = I * MC;

    uvl::Matrix<double, 3, 3> C(X + M);
    std::cout << "I" << std::endl;
    print(I);
    std::cout << "M" << std::endl;
    print(M);
    std::cout << "X" << std::endl;
    print(X);
    std::cout << "C" << std::endl;
    print(C);

    std::cout << "x" << std::endl;
    print(x);
    std::cout << "I * x" << std::endl;
    print(I * x);
    std::cout << "X  * x" << std::endl;
    print(X * x);

    uvl::MatrixBase swizzled = uvl::views::unary::SwizzleView<
        uvl::Matrix<double, 3, std::dynamic_extent>::view_type, 1, 0>(M.view());

    spdlog::info("M swizzled");
    print(swizzled);

    std::cout << "M.T * M" << std::endl;
    print(M.transpose() * M);

    uvl::MatrixBase C2 = uvl::views::nullary::ConstantView<double, 4, 4>(2);

    uvl::Matrix<double, 4, 4> M2 = C2;
    uvl::Matrix<double, std::dynamic_extent, std::dynamic_extent> M2d = C2;
    for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 4; ++k) {
            CHECK(C2(j, k) == 2);
            CHECK(M2(j, k) == 2);
            CHECK(M2d(j, k) == 2);
        }
    }

    std::random_device rd;

    uvl::MatrixBase R = uvl::views::nullary::uniform_random_view<double>(
        uvl::extents<4, 4>{}, -1, 1);

    M2 = R;
    print(M2);

    spdlog::info("and again");
    M2 = R;
    print(M2);

    spdlog::info("Integral stuff should come out");
    uvl::MatrixBase RI(uvl::views::nullary::uniform_random_view<int>(
        uvl::extents<4, 4>{}, 0, 40));
    M2 = RI.cast<double>();
    print(M2);

    spdlog::info("and again");
    M2 = RI.cast<double>();
    print(M2);

    spdlog::info("Normally distributed: ");
    uvl::MatrixBase RN(uvl::views::nullary::normal_random_view<double>(
        uvl::extents<4, 4>{}, 0, 20));
    uvl::Matrix MN = RN;
    print(MN);
    // CHECK(M2 == M2d);
}
