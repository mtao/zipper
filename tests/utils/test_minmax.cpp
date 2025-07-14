#include "../catch_include.hpp"


#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/minCoeff.hpp>
#include <zipper/utils/maxCoeff.hpp>


TEST_CASE("test_minMax_vector", "[storage][dense]") {
    
    zipper::Vector<double,3> x({0,1,2});
    zipper::Vector<double,3> y({1,2,0});


    CHECK(zipper::utils::minCoeff(x) == 0);
    CHECK(zipper::utils::maxCoeff(x) == 2);

    CHECK(zipper::utils::minCoeff(y) == 0);
    CHECK(zipper::utils::maxCoeff(y) == 2);


    {
    auto [m,c] = zipper::utils::minCoeffWithIndex(x);
    REQUIRE(c.size() == 1);
    CHECK(m == 0);
    CHECK(c[0] == 0);
    }
    {
    auto [m,c] = zipper::utils::maxCoeffWithIndex(x);
    REQUIRE(c.size() == 1);
    CHECK(m == 2);
    CHECK(c[0] == 2);
    }

    {
    auto [m,c] = zipper::utils::minCoeffWithIndex(y);
    REQUIRE(c.size() == 1);
    CHECK(m == 0);
    CHECK(c[0] == 2);
    }
    {
    auto [m,c] = zipper::utils::maxCoeffWithIndex(y);
    REQUIRE(c.size() == 1);
    CHECK(m == 2);
    CHECK(c[0] == 1);
    }

}
TEST_CASE("test_minMax_matrix", "[storage][dense]") {
    
    zipper::Matrix<double,3,3> x;
    x.col(0) = {0,1,2};
    x.col(1) = {-3,3,0};
    x.col(2) = {100,0,0};


    CHECK(zipper::utils::minCoeff(x) == -3);
    CHECK(zipper::utils::maxCoeff(x) == 100);



    {
    auto [m,c] = zipper::utils::minCoeffWithIndex(x);
    REQUIRE(c.size() == 2);
    CHECK(m == -3);
    CHECK(c == std::array<zipper::index_type,2>{{0,1}});
    }

    {
    auto [m,c] = zipper::utils::maxCoeffWithIndex(x);
    REQUIRE(c.size() == 2);
    CHECK(m == 100);
    CHECK(c == std::array<zipper::index_type,2>{{0,2}});
    }

}
