

#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
#include <iostream>
#include <uvl/ArrayBase.hpp>
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

    double l2 = x.norm<2>();
    double l1 = x.norm<1>();
    std::cout << "X norm: " << l1 << " " << l2 << std::endl;
    CHECK(l1 == 2 + 5 + 9);
    CHECK(l2 == std::sqrt(2 * 2 + 5 * 5 + 9 * 9));

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
    X.eval() + M.eval();
    X + M;

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

    for (int j = 0; j < 10; ++j) {
        spdlog::info("power trial {}", j);

        auto N = uvl::index_type((1 + j) * 10);
        uvl::Matrix At = uvl::views::nullary::uniform_random_view<double>(
            uvl::create_dextents(N, N), -1, 1);
        uvl::Matrix A = At.transpose() * At;
        uvl::Vector x = uvl::views::nullary::uniform_random_view<double>(
            uvl::create_dextents(N), -1, 1);

        for (int k = 0; k < j * 20; ++k) {
            // todo: assignment with copied data in case of aliasing
            x = (A * x).eval().normalized();
        }
        spdlog::info("x {}", x.extent(0));
        print(x);
        spdlog::info("Ax {} {}", A.extent(0), A.extent(1));
        print(A * x);
        double e = (A * x).norm();
        spdlog::info("Eigenvalue {}", e);
        print(e * x);
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }

    spdlog::info("Manipulating MN: ");
    print(MN);
    spdlog::warn("Full slice");
    print(MN.slice<uvl::full_extent_t, uvl::full_extent_t>());

    spdlog::warn("single row");

    auto slice = MN.slice<std::integral_constant<uvl::index_type, 1>,
                          uvl::full_extent_t>();
    spdlog::info("Slice rank {} extent {}", slice.extents().rank(),
                 slice.extent(0));
    print(slice);
    spdlog::info("Manipulating MN with access to slice: ");
    slice(0) = 2.0;
    slice(1) = 100000;
    slice(2) = 104000;
    slice(3) = 100003;

    spdlog::info("slice values: ");
    print(slice);
    spdlog::info("mn values: ");
    print(MN);
    slice = uvl::views::nullary::normal_random_view<double>(uvl::extents<4>{},
                                                            -200, 1e-2);
    spdlog::info("re-randomized slice: ");

    print(MN);

    spdlog::info("re-randomized slice: ");

    MN.row<std::integral_constant<uvl::index_type, 2>>() = slice;
    MN.col<std::integral_constant<uvl::index_type, 2>>() = slice;
    MN.col(2) = slice;
    MN.col(3) = uvl::views::nullary::normal_random_view<double>(
        uvl::extents<4>{}, -900, 1e-2);

    print(MN);

    spdlog::warn("Diagonal");

    print(MN.diagonal());

    auto slice2 = MN(std::integral_constant<uvl::index_type, 3>{},
                          uvl::full_extent_t{});
    //static_assert(std::is_same_v<std::decay_t<decltype(slice2)>, double>);
    slice2 = 100 * slice;
    spdlog::warn("slice2 assigns row 100 to be a multiple of row 1");
    print(MN);
    auto slice3 = MN(2,uvl::full_extent_t{});
    slice3 = 100 * slice2;
    spdlog::warn("slice3 assigns row 100 to be a multiple of row 2");
    print(MN);

    // MN.swizzle<1,0>() = MN;
}
