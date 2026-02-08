

#include <zipper/expression/unary/PartialTrace.hpp>

#include "fmt_include.hpp"
#include "catch_include.hpp"
#include <zipper/expression/nullary/Unit.hpp>
#include <iostream>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Form.hpp>
#include <zipper/MatrixBase.hxx>
#include <zipper/FormBase.hxx>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/Random.hpp>
#include <zipper/expression/unary/Swizzle.hpp>

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
                    //fmt::print("{} {} {} / {} {} {}\n", j, k, l, M.extents().static_extent(0),M.extents().static_extent(1),M.extents().static_extent(2));
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

    // CHECK(IM.slice(std::integral_constant<zipper::rank_type, 0>{},
    //                std::integral_constant<zipper::rank_type, 0>{},
    //                zipper::full_extent, zipper::full_extent) == M);

    // zipper::Tensor C ;
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

TEST_CASE("test_form_product", "[storage][tensor]") {
    // Test form-tensor contraction (dot product and higher-rank contractions)

    zipper::Vector<double, 3> O;
    O(0) = 1;
    O(1) = 1;
    O(2) = 1;

    zipper::Vector<double, 3> E0 = zipper::expression::nullary::unit_vector<double, 3>(0);
    zipper::Vector<double, 3> E1 = zipper::expression::nullary::unit_vector<double, 3>(1);
    zipper::Vector<double, 3> E2 = zipper::expression::nullary::unit_vector<double, 3>(2);

    zipper::Form<double, 3> D0 = zipper::expression::nullary::unit_vector<double, 3>(0);
    zipper::Form<double, 3> D1 = zipper::expression::nullary::unit_vector<double, 3>(1);
    zipper::Form<double, 3> D2 = zipper::expression::nullary::unit_vector<double, 3>(2);

    // Test that form * vector gives the Kronecker delta (identity)
    CHECK(double(D0 * E0) == 1.0);
    CHECK(double(D1 * E0) == 0.0);
    CHECK(double(D2 * E0) == 0.0);

    CHECK(double(D0 * E1) == 0.0);
    CHECK(double(D1 * E1) == 1.0);
    CHECK(double(D2 * E1) == 0.0);

    CHECK(double(D0 * E2) == 0.0);
    CHECK(double(D1 * E2) == 0.0);
    CHECK(double(D2 * E2) == 1.0);

    // Test dot product: form(O) * O = sum of components squared
    CHECK(double(O.as_form() * O) == Catch::Approx(3.0));

    // Test rank-2 form * rank-1 tensor contraction → rank-1 form
    // FormTensorProduct requires form_rank >= tensor_rank
    // (F * v)(i) = sum_j F(i,j) * v(j)
    zipper::Vector<double, 3> a =
        zipper::expression::nullary::normal_random_infinite<double>(0, 1);
    zipper::Vector<double, 3> b =
        zipper::expression::nullary::normal_random_infinite<double>(0, 1);

    zipper::Vector<double, 3> x =
        zipper::expression::nullary::normal_random_infinite<double>(0, 1);
    zipper::Vector<double, 3> y =
        zipper::expression::nullary::normal_random_infinite<double>(0, 1);

    // Build a rank-2 form from wedge product: F = x^ ^ y^
    auto F_xy = (x.as_form() ^ y.as_form()).eval();
    static_assert(decltype(F_xy)::extents_type::rank() == 2);

    // Test wedge product antisymmetry: (x^ ^ y^)(i,j) = x(i)*y(j) - x(j)*y(i)
    for (zipper::index_type i = 0; i < 3; ++i) {
        for (zipper::index_type j = 0; j < 3; ++j) {
            double expected_val = x(i) * y(j) - x(j) * y(i);
            CHECK(F_xy(i, j) == Catch::Approx(expected_val));
            // Antisymmetry: w(i,j) = -w(j,i)
            CHECK(F_xy(i, j) == Catch::Approx(-F_xy(j, i)));
        }
    }

    // Contract rank-2 form with rank-1 tensor: result is rank-1 form
    // (F_xy * a)(i) = sum_j F_xy(i,j) * a(j)
    auto F_xy_a = (F_xy * a).eval();
    static_assert(decltype(F_xy_a)::extents_type::rank() == 1);

    for (zipper::index_type i = 0; i < 3; ++i) {
        double expected_val = 0.0;
        for (zipper::index_type j = 0; j < 3; ++j) {
            expected_val += F_xy(i, j) * a(j);
        }
        CHECK(F_xy_a(i) == Catch::Approx(expected_val));
    }

    // Chained contraction: (F_xy * a) * b should give scalar
    // sum_i (sum_j F_xy(i,j) * a(j)) * b(i)
    double chained = double(F_xy_a * b);
    double expected_chained = 0.0;
    for (zipper::index_type i = 0; i < 3; ++i) {
        for (zipper::index_type j = 0; j < 3; ++j) {
            expected_chained += F_xy(i, j) * a(j) * b(i);
        }
    }
    CHECK(chained == Catch::Approx(expected_chained));

    // Test rank-1 form * rank-2 tensor → rank-1 tensor
    // When form_rank < tensor_rank, result is a TensorBase
    // (f * T)(j) = sum_i f(i) * T(i, j)
    auto AB = (a.as_tensor() * b.as_tensor()).eval();
    static_assert(decltype(AB)::extents_type::rank() == 2);

    auto xf = x.as_form();
    auto xf_AB = xf * AB;
    static_assert(std::decay_t<decltype(xf_AB)>::extents_type::rank() == 1);

    // (x^ * (a⊗b))(j) = sum_i x(i) * a(i) * b(j) = (x·a) * b(j)
    double x_dot_a = double(x.as_form() * a);
    for (zipper::index_type j = 0; j < 3; ++j) {
        CHECK(double(xf_AB(j)) == Catch::Approx(x_dot_a * b(j)));
    }
}
