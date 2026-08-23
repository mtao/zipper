#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/max_coeff.hpp>
#include <zipper/utils/mean_coeff.hpp>
#include <zipper/utils/min_coeff.hpp>

#include "../catch_include.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

TEST_CASE("test_minMax_vector", "[storage][dense]") {
    zipper::Vector<double, 3> x({0, 1, 2});
    zipper::Vector<double, 3> y({1, 2, 0});

    CHECK(zipper::utils::minCoeff(x) == 0);
    CHECK(zipper::utils::maxCoeff(x) == 2);
    CHECK(zipper::utils::meanCoeff(x) == 1);

    CHECK(zipper::utils::minCoeff(y) == 0);
    CHECK(zipper::utils::maxCoeff(y) == 2);
    CHECK(zipper::utils::meanCoeff(y) == 1);

    {
        auto [m, c] = zipper::utils::minCoeffWithIndex(x);
        REQUIRE(c.size() == 1);
        CHECK(m == 0);
        CHECK(c[0] == 0);
    }
    {
        auto [m, c] = zipper::utils::maxCoeffWithIndex(x);
        REQUIRE(c.size() == 1);
        CHECK(m == 2);
        CHECK(c[0] == 2);
    }

    {
        auto [m, c] = zipper::utils::minCoeffWithIndex(y);
        REQUIRE(c.size() == 1);
        CHECK(m == 0);
        CHECK(c[0] == 2);
    }
    {
        auto [m, c] = zipper::utils::maxCoeffWithIndex(y);
        REQUIRE(c.size() == 1);
        CHECK(m == 2);
        CHECK(c[0] == 1);
    }
}
TEST_CASE("test_minMax_matrix", "[storage][dense]") {
    zipper::Matrix<double, 3, 3> x;
    x.col(0) = {0, 1, 2};
    x.col(1) = {-3, 3, 0};
    x.col(2) = {100, 4, 1};

    CHECK(x.as_array().sum() == 108);
    CHECK(zipper::utils::minCoeff(x) == -3);
    CHECK(zipper::utils::maxCoeff(x) == 100);
    CHECK(zipper::utils::meanCoeff(x) == 12);

    {
        auto [m, c] = zipper::utils::minCoeffWithIndex(x);
        REQUIRE(c.size() == 2);
        CHECK(m == -3);
        CHECK(c == std::array<zipper::index_type, 2>{{0, 1}});
    }

    {
        auto [m, c] = zipper::utils::maxCoeffWithIndex(x);
        REQUIRE(c.size() == 2);
        CHECK(m == 100);
        CHECK(c == std::array<zipper::index_type, 2>{{0, 2}});
    }
}

TEST_CASE("max_coeff_with_index_initializes_index_for_nan", "[utils][max]") {
    zipper::Vector<double, 2> values{
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()};

    const auto [value, index] = zipper::utils::maxCoeffWithIndex(values);

    CHECK(std::isnan(value));
    CHECK(index[0] == 0);
}

TEST_CASE("max_coeff_rejects_empty_inputs", "[utils][max]") {
    zipper::VectorX<double> values(0);
    CHECK_THROWS_AS(zipper::utils::maxCoeff(values), std::invalid_argument);
}
