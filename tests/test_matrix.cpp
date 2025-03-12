

#include <iostream>
#include <spdlog/spdlog.h>

#include <catch2/catch_all.hpp>
#include <uvl/Matrix.hpp>
//#include <uvl/Vector.hpp>

TEST_CASE("test_all_extents", "[storage][dense]") {
    uvl::Matrix<double, 3, 3> L;
    uvl::Matrix<double, 3, std::dynamic_extent> M(3);
    //uvl::Vector<double, 3> x;

   // x(0) = 2;
   // x(1) = 5;
   // x(3) = 9;

    L(0,0) = 1;
    L(1,0) = 0;
    L(2,0) = 0;
    L(0,1) = 0;
    L(1,1) = 1;
    L(2,1) = 0;
    L(0,2) = 0;
    L(1,2) = 0;
    L(2,2) = 1;

    M(0,0) = 2;
    M(1,0) = 3;
    M(2,0) = 1;
    M(0,1) = 0;
    M(1,1) = 9;
    M(2,1) = 8;
    M(0,2) = 6;
    M(1,2) = 8;
    M(2,2) = 6;

    auto MC = M.cast<double>();
    auto X = L * MC;

    uvl::Matrix<double, 3, 3> C(X + M);
    std::cout << "L" << std::endl;
    for(int j = 0; j < 3; ++j) {
        for(int k = 0; k < 3; ++k) {
            std::cout << L(j,k) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "M" << std::endl;
    for(int j = 0; j < 3; ++j) {
        for(int k = 0; k < 3; ++k) {
            std::cout << M(j,k) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "X" << std::endl;
    for(int j = 0; j < 3; ++j) {
        for(int k = 0; k < 3; ++k) {
            std::cout << X(j,k) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "C" << std::endl;
    for(int j = 0; j < 3; ++j) {
        for(int k = 0; k < 3; ++k) {
            std::cout << C(j,k) << " ";
        }
        std::cout << std::endl;
    }
}
