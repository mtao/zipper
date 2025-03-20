

#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
#include <iostream>
#include <uvl/Tensor.hpp>
#include <uvl/TensorBase.hpp>
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
TEST_CASE("test_all_extents", "[storage][dense]") {
    uvl::Tensor<double, 3, 3> I =
        uvl::TensorBase(uvl::views::nullary::IdentityView<double, 3, 3>{});
    uvl::Tensor<double, 3, std::dynamic_extent> M(3);
    uvl::Tensor<double, 3> x;

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

    // uvl::Tensor C ;
}
