

#include <iostream>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/MatrixBase.hxx>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/Random.hpp>
#include <zipper/expression/unary/PartialTrace.hpp>

#include "../../catch_include.hpp"

namespace {
void print(auto const& M) {
    constexpr static zipper::rank_type rank =
        std::decay_t<decltype(M)>::extents_type::rank();

    if constexpr (rank == 1) {
        for (zipper::index_type j = 0; j < M.extent(0); ++j) {
            std::cout << M(j) << " ";
            std::cout << std::endl;
        }
    } else if constexpr (rank == 2) {
        for (zipper::index_type j = 0; j < M.extent(0); ++j) {
            for (zipper::index_type k = 0; k < M.extent(1); ++k) {
                std::cout << M(j, k) << " ";
            }
            std::cout << std::endl;
        }
    } else if constexpr (rank == 3) {
        for (zipper::index_type j = 0; j < M.extent(0); ++j) {
            for (zipper::index_type k = 0; k < M.extent(1); ++k) {
                for (zipper::index_type l = 0; l < M.extent(2); ++l) {
                    std::cout << M(j, k, l) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "-----" << std::endl;
        }
    } else if constexpr (rank == 4) {
        for (zipper::index_type j = 0; j < M.extent(0); ++j) {
            for (zipper::index_type k = 0; k < M.extent(1); ++k) {
                for (zipper::index_type l = 0; l < M.extent(2); ++l) {
                    for (zipper::index_type m = 0; m < M.extent(3); ++m) {
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
    zipper::Tensor<double, 3, 3> I = zipper::expression::nullary::Identity<double>{};
    zipper::Tensor<double, 3, std::dynamic_extent> M(3);
    zipper::Tensor<double, 3> x;

    zipper::Tensor<double, 3, 3, 3> J =
        zipper::expression::nullary::Constant<double>{6};
    //fmt::print("Constant tensor from infinite view\n");
    print(J);

    M = zipper::expression::nullary::normal_random_infinite<double>(0, 1);

    x = zipper::expression::nullary::normal_random_infinite<double>(10, 1);

    print(M);

    x(0) = 2;
    x(1) = 5;
    x(2) = 9;

    //fmt::print("Prod of matrix vector\n");
    print(M * x);
    //fmt::print("Prod of matrix Matrix identity\n");
    print(I * M);
    auto IM = I * M;
    static_assert(decltype(I)::extents_type::rank() == 2);
    static_assert(decltype(M)::extents_type::rank() == 2);
    static_assert(decltype(IM)::extents_type::rank() == 4);
}

TEST_CASE("test_product", "[storage][tensor]") {
    // Test that matrix product can be computed via tensor product + partial trace
    zipper::Tensor<double, 3, 3> I = zipper::expression::nullary::Identity<double>{};
    zipper::Tensor<double, 3, 3> M =
        zipper::expression::nullary::normal_random_infinite<double>(0, 1);

    zipper::Tensor<double, 3, 3> N =
        zipper::expression::nullary::normal_random_infinite<double>(0, 1);

    // Matrix product via MatrixBase operator*
    auto mM = zipper::as_matrix(M);
    auto mN = zipper::as_matrix(N);
    auto MN_matrix = (mM * mN).eval();

    // Matrix product via tensor product + partial trace
    auto TP = M * N;
    static_assert(decltype(TP)::extents_type::rank() == 4);

    using TP_type = std::decay_t<decltype(TP)>::expression_type;
    zipper::expression::unary::PartialTrace<const TP_type, 1, 2> pt(TP.expression());
    zipper::Matrix<double, 3, 3> MN_tensor = pt;

    // Both methods should give the same result
    for (zipper::index_type j = 0; j < 3; ++j) {
        for (zipper::index_type k = 0; k < 3; ++k) {
            CHECK(MN_matrix(j, k) == Catch::Approx(MN_tensor(j, k)));
        }
    }
}
