

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
   //
TEST_CASE("test_assignment", "[matrix][storage][dense]") {
    uvl::Matrix<double, 3, std::dynamic_extent> M(3);
    uvl::Matrix<double, 3, 5> N(3, 5);
    uvl::Matrix<double, std::dynamic_extent, std::dynamic_extent> O(2, 2);
    uvl::Vector<double, 3> x;

    O(0, 0) = 0;
    O(1, 0) = 1;
    O(0, 1) = 3;
    O(1, 1) = 4;

    CHECK(O(0, 0) == 0);
    CHECK(O(1, 0) == 1);
    CHECK(O(0, 1) == 3);
    CHECK(O(1, 1) == 4);

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

    N(0, 0) = 0;
    N(1, 0) = 1;
    N(2, 0) = 2;
    N(0, 1) = 3;
    N(1, 1) = 4;
    N(2, 1) = 5;
    N(0, 2) = 6;
    N(1, 2) = 7;
    N(2, 2) = 8;
    N(0, 3) = 9;
    N(1, 3) = 10;
    N(2, 3) = 11;
    N(0, 4) = 12;
    N(1, 4) = 13;
    N(2, 4) = 14;

    CHECK(x(0) == 2);
    CHECK(x(1) == 5);
    CHECK(x(2) == 9);
    CHECK(M(0, 0) == 0);
    CHECK(M(1, 0) == 1);
    CHECK(M(2, 0) == 2);
    CHECK(M(0, 1) == 3);
    CHECK(M(1, 1) == 4);
    CHECK(M(2, 1) == 5);
    CHECK(M(0, 2) == 6);
    CHECK(M(1, 2) == 7);
    CHECK(M(2, 2) == 8);
    CHECK(N(0, 0) == 0);
    CHECK(N(1, 0) == 1);
    CHECK(N(2, 0) == 2);
    CHECK(N(0, 1) == 3);
    CHECK(N(1, 1) == 4);
    CHECK(N(2, 1) == 5);
    CHECK(N(0, 2) == 6);
    CHECK(N(1, 2) == 7);
    CHECK(N(2, 2) == 8);
    CHECK(N(0, 3) == 9);
    CHECK(N(1, 3) == 10);
    CHECK(N(2, 3) == 11);
    CHECK(N(0, 4) == 12);
    CHECK(N(1, 4) == 13);
    CHECK(N(2, 4) == 14);
}

TEST_CASE("test_transpose", "[matrix][storage][dense]") {
    uvl::Matrix<double, 3, std::dynamic_extent> M(3);
    uvl::Matrix<double, 3, 5> N(3, 5);

    for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
            M(j, k) = 3 * j + k;
        }
    }

    for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 5; ++k) {
            N(j, k) = 5 * j + k;
        }
    }

    REQUIRE(M.transpose().extent(0) == 3);
    REQUIRE(M.transpose().extent(1) == 3);
    for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
            CHECK(M.transpose()(k, j) == 3 * j + k);
        }
    }

    REQUIRE(N.transpose().extent(0) == 5);
    REQUIRE(N.transpose().extent(1) == 3);

    REQUIRE(N.transpose().extents().static_extent(0) == 5);
    REQUIRE(N.transpose().extents().static_extent(1) == 3);
    for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 5; ++k) {
            CHECK(N.transpose()(k, j) == 5 * j + k);
        }
    }
    uvl::Matrix MTM = M.transpose() * M;
    REQUIRE(MTM.transpose().extent(0) == 3);
    REQUIRE(MTM.transpose().extent(1) == 3);

    REQUIRE(MTM.transpose().extents().static_extent(0) == std::dynamic_extent);
    REQUIRE(MTM.transpose().extents().static_extent(1) == std::dynamic_extent);
    print(MTM);
    CHECK(MTM(0, 0) == 45);
    CHECK(MTM(1, 0) == 54);
    CHECK(MTM(2, 0) == 63);
    CHECK(MTM(0, 1) == 54);
    CHECK(MTM(1, 1) == 66);
    CHECK(MTM(2, 1) == 78);
    CHECK(MTM(0, 2) == 63);
    CHECK(MTM(1, 2) == 78);
    CHECK(MTM(2, 2) == 93);

    uvl::Matrix MMT = M * M.transpose();
    REQUIRE(MMT.transpose().extent(0) == 3);
    REQUIRE(MMT.transpose().extent(1) == 3);

    REQUIRE(MMT.transpose().extents().static_extent(0) == 3);
    REQUIRE(MMT.transpose().extents().static_extent(1) == 3);
    print(MMT);
    CHECK(MMT(0, 0) == 5);
    CHECK(MMT(1, 0) == 14);
    CHECK(MMT(2, 0) == 23);
    CHECK(MMT(0, 1) == 14);
    CHECK(MMT(1, 1) == 50);
    CHECK(MMT(2, 1) == 86);
    CHECK(MMT(0, 2) == 23);
    CHECK(MMT(1, 2) == 86);
    CHECK(MMT(2, 2) == 149);
}

TEST_CASE("test_identity", "[matrix][identity]") {
    auto check_identity = [](const auto& i) {
        for (uvl::index_type j = 0; j < i.extent(0); ++j) {
            for (uvl::index_type k = 0; k < i.extent(1); ++k) {
                if (j == k) {
                    CHECK(i(j, k) == 1);
                } else {
                    CHECK(i(j, k) == 0);
                }
            }
        }
    };
    uvl::Matrix<double, 3, 3> I =
        uvl::MatrixBase(uvl::views::nullary::IdentityView<double, 3, 3>{});
    check_identity(I);
    check_identity(
        uvl::MatrixBase(uvl::views::nullary::IdentityView<double, 3, 3>{}));
    CHECK(
        (I ==
         uvl::MatrixBase(
             uvl::views::nullary::IdentityView<double, std::dynamic_extent, 3>{
                 3}))
            .all());
    CHECK(
        (I == uvl::MatrixBase(
                  uvl::views::nullary::IdentityView<double, std::dynamic_extent,
                                                    std::dynamic_extent>{3, 3}))
            .all());

    check_identity(uvl::MatrixBase(
        uvl::views::nullary::IdentityView<double, std::dynamic_extent,
                                          std::dynamic_extent>{20, 20}));

    uvl::Matrix<double, 3, 3> M =
        uvl::views::nullary::uniform_random_view<double>(uvl::extents<3, 3>{},
                                                         0, 5);

    uvl::Matrix<double, 3, 3> MI = I * M;
    uvl::Matrix<double, 3, 3> IM = M * I;
    CHECK((M == MI).all());
    CHECK((M == IM).all());
}

TEST_CASE("test_identity", "[matrix][vector][lift]") {
    uvl::Vector<double, 3> x;
    x(0) = 2;
    x(1) = 5;
    x(2) = 9;
    auto colMat = x.swizzle<uvl::MatrixBase, 0, std::dynamic_extent>();
    auto rowMat = x.swizzle<uvl::MatrixBase, std::dynamic_extent, 0>();
    spdlog::info("Vector x is: {}");
    REQUIRE(x.extents().rank() == 1);
    REQUIRE(x.extent(0) == 3);
    CHECK(x(0) == 2);
    CHECK(x(1) == 5);
    CHECK(x(2) == 9);
    print(x);
    spdlog::info("Vector x as colmat:");
    REQUIRE(colMat.extents().rank() == 2);
    REQUIRE(colMat.extent(0) == 3);
    REQUIRE(colMat.extent(1) == 1);

    CHECK(colMat(0, 0) == 2);
    CHECK(colMat(1, 0) == 5);
    CHECK(colMat(2, 0) == 9);
    print(colMat);
    spdlog::info("Vector x as rowmat:");
    REQUIRE(rowMat.extents().rank() == 2);
    REQUIRE(rowMat.extent(0) == 1);
    REQUIRE(rowMat.extent(1) == 3);
    CHECK(rowMat(0, 0) == 2);
    CHECK(rowMat(0, 1) == 5);
    CHECK(rowMat(0, 2) == 9);
    print(rowMat);
}

TEST_CASE("test_trace", "[matrix][storage][dense]") {
    uvl::Matrix<double, 3, 3> N(3, 5);
    N = uvl::views::nullary::uniform_random_view<double>(uvl::extents<3, 3>{},
                                                         -1, 1);
    N.diagonal() = uvl::views::nullary::ConstantView<double, 3>(0.0);
    CHECK(N.trace() == 0);

    N.diagonal() = uvl::views::nullary::ConstantView<double, 3>(1.0);
    print(N);
    CHECK(N.trace() == 3);
    N(0, 0) = 2;
    CHECK(N.trace() == 4);
    N(1, 1) = 2;
    CHECK(N.trace() == 5);
    N(2, 2) = 2;
    CHECK(N.trace() == 6);
}
/*
TEST_CASE("test_all_extents", "[storage][dense]") {



    double l2 = x.norm<2>();
    double l1 = x.norm<1>();
    std::cout << "X norm: " << l1 << " " << l2 << std::endl;
    CHECK(l1 == 2 + 5 + 9);
    CHECK(l2 == std::sqrt(2 * 2 + 5 * 5 + 9 * 9));


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
    MN.col(3) = uvl::views::nullary::normal_random_view<double>(
        uvl::extents<4>{}, -900, 1e-2);
    slice = uvl::views::nullary::normal_random_view<double>(uvl::extents<4>{},
                                                            -200, 1e-2);

}
*/
