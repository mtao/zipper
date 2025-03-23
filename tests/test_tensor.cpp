

#include <uvl/views/unary/PartialTraceView.hpp>
#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
#include <iostream>
#include <uvl/Tensor.hpp>
#include <uvl/TensorBase.hpp>
#include <uvl/MatrixBase.hpp>
#include <uvl/views/nullary/ConstantView.hpp>
#include <uvl/views/nullary/IdentityView.hpp>
#include <uvl/views/nullary/RandomView.hpp>
#include <uvl/views/unary/SwizzleView.hpp>
// #include <uvl/Vector.hpp>

namespace {

void print(uvl::concepts::TensorBaseDerived auto const& M) {
    constexpr static uvl::rank_type rank =
        std::decay_t<decltype(M)>::extents_type::rank();

    if constexpr (rank == 1) {
        for (uvl::index_type j = 0; j < M.extent(0); ++j) {
            std::cout << M(j) << " ";
            std::cout << std::endl;
        }
    } else if constexpr (rank == 2) {
        for (uvl::index_type j = 0; j < M.extent(0); ++j) {
            for (uvl::index_type k = 0; k < M.extent(1); ++k) {
                std::cout << M(j, k) << " ";
            }
            std::cout << std::endl;
        }
    } else if constexpr (rank == 3) {
        for (uvl::index_type j = 0; j < M.extent(0); ++j) {
            for (uvl::index_type k = 0; k < M.extent(1); ++k) {
                for (uvl::index_type l = 0; l < M.extent(2); ++l) {
                    std::cout << M(j, k, l) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "-----" << std::endl;
        }
    } else if constexpr (rank == 4) {
        for (uvl::index_type j = 0; j < M.extent(0); ++j) {
            for (uvl::index_type k = 0; k < M.extent(1); ++k) {
                for (uvl::index_type l = 0; l < M.extent(2); ++l) {
                    for (uvl::index_type m = 0; m < M.extent(3); ++m) {
                        std::cout << M(j, k, l, m) << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << "-----" << std::endl;
            }
            std::cout << "=====" << std::endl;
        }
    }
}
}  // namespace
TEST_CASE("test_tensor_product", "[storage][dense]") {
    uvl::Tensor<double, 3, 3> I = uvl::views::nullary::IdentityView<double>{};
    uvl::Tensor<double, 3, std::dynamic_extent> M(3);
    uvl::Tensor<double, 3> x;

    uvl::Tensor<double, 3, 3, 3> J =
        uvl::views::nullary::ConstantView<double>{6};
    spdlog::info("Constant tensor from infinite view");
    print(J);

    M = uvl::views::nullary::normal_random_infinite_view<double>(0, 1);

    x = uvl::views::nullary::normal_random_infinite_view<double>(10, 1);

    print(M);

    x(0) = 2;
    x(1) = 5;
    x(2) = 9;

    spdlog::info("Prod of matrix vector");
    print(M * x);
    spdlog::info("Prod of matrix Matrix identity");
    print(I * M);
    auto IM = I * M;
    static_assert(decltype(I)::extents_type::rank() == 2);
    static_assert(decltype(M)::extents_type::rank() == 2);
    static_assert(decltype(IM)::extents_type::rank() == 4);

    // CHECK(IM.slice(std::integral_constant<uvl::rank_type, 0>{},
    //                std::integral_constant<uvl::rank_type, 0>{},
    //                uvl::full_extent, uvl::full_extent) == M);

    // uvl::Tensor C ;
}

TEST_CASE("test_product", "[storage][tensor]") {
    uvl::Tensor<double, 3, 3> I = uvl::views::nullary::IdentityView<double>{};
    uvl::Tensor<double, 3, 3> M =
        uvl::views::nullary::normal_random_infinite_view<double>(0, 1);

    uvl::Tensor<double, 3, 3> N =
        uvl::views::nullary::normal_random_infinite_view<double>(0, 1);


    uvl::MatrixBase mM = M.view();
    uvl::MatrixBase mN = N.view();

    auto TP = M * N;

    static_assert(decltype(TP)::extents_type::rank() == 4);
    uvl::MatrixBase MN_tensor =
        uvl::views::unary::PartialTraceView<std::decay_t<decltype(TP.view())>,1,2>(TP.view());
    CHECK(mM*mN == MN_tensor);



}
