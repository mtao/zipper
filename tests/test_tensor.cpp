

#include <zipper/views/unary/PartialTraceView.hpp>
#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
#include <zipper/views/nullary/UnitView.hpp>
#include <iostream>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/MatrixBase.hpp>
#include <zipper/FormBase.hpp>
#include <zipper/views/nullary/ConstantView.hpp>
#include <zipper/views/nullary/IdentityView.hpp>
#include <zipper/views/nullary/RandomView.hpp>
#include <zipper/views/unary/SwizzleView.hpp>
// #include <zipper/Vector.hpp>

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
                    spdlog::info("{} {} {} / {} {} {}", j, k, l, M.extents().static_extent(0),M.extents().static_extent(1),M.extents().static_extent(2));
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
    zipper::Tensor<double, 3, 3> I = zipper::views::nullary::IdentityView<double>{};
    zipper::Tensor<double, 3, std::dynamic_extent> M(3);
    zipper::Tensor<double, 3> x;

    zipper::Tensor<double, 3, 3, 3> J =
        zipper::views::nullary::ConstantView<double>{6};
    spdlog::info("Constant tensor from infinite view");
    print(J);

    M = zipper::views::nullary::normal_random_infinite_view<double>(0, 1);

    x = zipper::views::nullary::normal_random_infinite_view<double>(10, 1);

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

    // CHECK(IM.slice(std::integral_constant<zipper::rank_type, 0>{},
    //                std::integral_constant<zipper::rank_type, 0>{},
    //                zipper::full_extent, zipper::full_extent) == M);

    // zipper::Tensor C ;
}

    /*
TEST_CASE("test_product", "[storage][tensor]") {
    zipper::Tensor<double, 3, 3> I = zipper::views::nullary::IdentityView<double>{};
    zipper::Tensor<double, 3, 3> M =
        zipper::views::nullary::normal_random_infinite_view<double>(0, 1);

    zipper::Tensor<double, 3, 3> N =
        zipper::views::nullary::normal_random_infinite_view<double>(0, 1);


    zipper::MatrixBase mM = M.view();
    zipper::MatrixBase mN = N.view();

    auto TP = M * N;

    static_assert(decltype(TP)::extents_type::rank() == 4);
    zipper::MatrixBase MN_tensor =
        zipper::views::unary::PartialTraceView<std::decay_t<decltype(TP.view())>,1,2>(TP.view());
    CHECK(mM*mN == MN_tensor);



}
TEST_CASE("test_form_product", "[storage][tensor]") {

    zipper::Vector<double,3> O;
    O(0) = 1;
    O(1) = 1;
    O(2) = 1;

    zipper::Vector<double,3> E0 = zipper::views::nullary::unit_vector<double,3>(0);
    zipper::Vector<double,3> E1 = zipper::views::nullary::unit_vector<double,3>(1);
    zipper::Vector<double,3> E2 = zipper::views::nullary::unit_vector<double,3>(2);

    zipper::Form<double,3> D0 = zipper::views::nullary::unit_vector<double,3>(0);
    zipper::Form<double,3> D1 = zipper::views::nullary::unit_vector<double,3>(1);
    zipper::Form<double,3> D2 = zipper::views::nullary::unit_vector<double,3>(2);

    spdlog::warn("Identity matrix products");
    auto E = (D0*E0).eval();
    spdlog::info("E extents: {} {}", E.extents(), double(E));

    spdlog::info("{} {} {}", 
            double(D0*E0),
            double(D1*E0),
            double(D2*E0));

    spdlog::info("{} {} {}", 
            double(D0*E1),
            double(D1*E1),
            double(D2*E1));


    spdlog::info("{} {} {}", 
            double(D0*E2),
            double(D1*E2),
            double(D2*E2));

    zipper::Vector<double,3> T;
    T(0) = 0;
    T(1) = 1;
    T(2) = 1;
    zipper::Vector<double,3> a = 
        zipper::views::nullary::normal_random_infinite_view<double>(0, 1);
    zipper::Vector<double,3> b = 
        zipper::views::nullary::normal_random_infinite_view<double>(0, 1);
    zipper::Vector<double,3> c = 
        zipper::views::nullary::normal_random_infinite_view<double>(0, 1);

    zipper::Vector<double,3> x = 
        zipper::views::nullary::normal_random_infinite_view<double>(0, 1);
    zipper::Vector<double,3> y = 
        zipper::views::nullary::normal_random_infinite_view<double>(0, 1);
    zipper::Vector<double,3> z = 
        zipper::views::nullary::normal_random_infinite_view<double>(0, 1);


    spdlog::warn("Dot product {}", double(O.as_form() * O));


    auto AB = (a.as_tensor() * b.as_tensor()).eval();

    auto XY= (x.as_form() * y.as_form()).eval();

    spdlog::warn("Tensor:");
    print(AB);
    spdlog::warn("Form:");
    print(XY);
    spdlog::info("Contracted should be: {} got {}",
            double(y.as_form() * b.as_tensor()) * 
            double(x.as_form() * c.as_tensor())
            ,double(XY * AB));

    spdlog::warn("PRODUCT:");
    print(XY * a);
    spdlog::warn("PRODUCT PRODUCT  should be near 0:");
    double M = ((XY * O).eval() * O).eval();
    spdlog::info("Otput: {}", M);

    spdlog::warn("PRODUCT PRODUCT  should be otherwise:");
    double M2 = ((XY * O).eval() * T).eval();
    spdlog::info("Otput: {}", M2);
    //print();



    auto ABC = a.as_tensor() * b.as_tensor() * c.as_tensor();

    auto XYZ = x.as_form() * y.as_form() * z.as_form();

    spdlog::warn("Tensor:");
    print(ABC);
    spdlog::warn("Form:");
    print(XYZ);

    spdlog::info("Contracted should be: {} got {}",
            double(z.as_form() * a.as_tensor()) * 
            double(y.as_form() * b.as_tensor()) * 
            double(x.as_form() * c.as_tensor())
            ,double(XYZ * ABC));

}
*/
