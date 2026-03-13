
#include <zipper/Form.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/eigen/power_method.hpp>

#include "../catch_include.hpp"

using namespace zipper;

TEST_CASE("power_method static 2x2", "[eigen][power_method]") {
    // Matrix with known eigenvalues:
    // [[2, 1], [1, 2]] has eigenvalues 3 and 1
    // Dominant eigenvalue is 3, eigenvector is [1/sqrt(2), 1/sqrt(2)]
    Matrix<double, 2, 2> M{{2.0, 1.0}, {1.0, 2.0}};

    auto result = utils::eigen::power_method(M);

    REQUIRE(result.has_value());
    CHECK(result->eigenvalue == Catch::Approx(3.0).epsilon(1e-8));
    // Eigenvector should be proportional to [1, 1]
    CHECK(std::abs(result->eigenvector(0)) ==
          Catch::Approx(std::abs(result->eigenvector(1))).epsilon(1e-8));
    // Should be normalized
    CHECK(result->eigenvector.norm() == Catch::Approx(1.0).epsilon(1e-8));
}

TEST_CASE("power_method static 3x3", "[eigen][power_method]") {
    // Diagonal matrix: eigenvalues are on the diagonal
    // Dominant eigenvalue is 5.0
    Matrix<double, 3, 3> M{
        {5.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 1.0},
    };

    auto result = utils::eigen::power_method(M);

    REQUIRE(result.has_value());
    CHECK(result->eigenvalue == Catch::Approx(5.0).epsilon(1e-8));
    // Eigenvector should be [1, 0, 0] (or [-1, 0, 0])
    CHECK(std::abs(result->eigenvector(0)) == Catch::Approx(1.0).epsilon(1e-8));
    CHECK(std::abs(result->eigenvector(1)) == Catch::Approx(0.0).margin(1e-8));
    CHECK(std::abs(result->eigenvector(2)) == Catch::Approx(0.0).margin(1e-8));
}

TEST_CASE("power_method symmetric 3x3", "[eigen][power_method]") {
    // Symmetric matrix with eigenvalues 6, 3, 3
    // [[4, 1, 1], [1, 4, 1], [1, 1, 4]] has eigenvalues 6, 3, 3
    Matrix<double, 3, 3> M{
        {4.0, 1.0, 1.0},
        {1.0, 4.0, 1.0},
        {1.0, 1.0, 4.0},
    };

    auto result = utils::eigen::power_method(M);

    REQUIRE(result.has_value());
    CHECK(result->eigenvalue == Catch::Approx(6.0).epsilon(1e-8));
    // Eigenvector for eigenvalue 6 is proportional to [1, 1, 1]
    double inv_sqrt3 = 1.0 / std::sqrt(3.0);
    CHECK(std::abs(result->eigenvector(0)) == Catch::Approx(inv_sqrt3).epsilon(1e-6));
    CHECK(std::abs(result->eigenvector(1)) == Catch::Approx(inv_sqrt3).epsilon(1e-6));
    CHECK(std::abs(result->eigenvector(2)) == Catch::Approx(inv_sqrt3).epsilon(1e-6));
}

TEST_CASE("power_method dynamic matrix", "[eigen][power_method]") {
    MatrixXX<double> M(2, 2);
    M(0, 0) = 3.0;
    M(0, 1) = 1.0;
    M(1, 0) = 1.0;
    M(1, 1) = 3.0;

    auto result = utils::eigen::power_method(M);

    REQUIRE(result.has_value());
    CHECK(result->eigenvalue == Catch::Approx(4.0).epsilon(1e-8));
}

TEST_CASE("power_method negative dominant eigenvalue", "[eigen][power_method]") {
    // [[-3, 0], [0, 1]] has eigenvalues -3 and 1
    // Dominant eigenvalue by magnitude is -3
    Matrix<double, 2, 2> M{{-3.0, 0.0}, {0.0, 1.0}};

    auto result = utils::eigen::power_method(M);

    REQUIRE(result.has_value());
    CHECK(result->eigenvalue == Catch::Approx(-3.0).epsilon(1e-8));
}

TEST_CASE("power_method identity", "[eigen][power_method]") {
    // Identity matrix: all eigenvalues are 1
    // Power method may not converge nicely when eigenvalues have
    // equal magnitude, but it should still return eigenvalue ~1
    Matrix<double, 2, 2> M{{1.0, 0.0}, {0.0, 1.0}};

    auto result = utils::eigen::power_method(M);

    // May or may not converge for repeated eigenvalues, but if it
    // does return a value the eigenvalue should be ~1
    if (result.has_value()) {
        CHECK(result->eigenvalue == Catch::Approx(1.0).epsilon(1e-6));
    } else {
        // Acceptable: power method doesn't converge for repeated eigenvalues
        CHECK_THAT(result.error().message,
                   Catch::Matchers::ContainsSubstring("converge"));
    }
}

TEST_CASE("power_method non-square returns error", "[eigen][power_method]") {
    MatrixXX<double> M(2, 3);
    auto result = utils::eigen::power_method(M);

    REQUIRE_FALSE(result.has_value());
    CHECK_THAT(result.error().message,
               Catch::Matchers::ContainsSubstring("square"));
}

TEST_CASE("power_method empty returns error", "[eigen][power_method]") {
    MatrixXX<double> M(0, 0);
    auto result = utils::eigen::power_method(M);

    REQUIRE_FALSE(result.has_value());
    CHECK_THAT(result.error().message,
               Catch::Matchers::ContainsSubstring("non-empty"));
}

TEST_CASE("power_method with initial guess", "[eigen][power_method]") {
    // [[2, 1], [1, 2]] has dominant eigenvalue 3, eigenvector [1,1]/sqrt(2)
    Matrix<double, 2, 2> M{{2.0, 1.0}, {1.0, 2.0}};

    // Provide an initial guess close to the dominant eigenvector
    Vector<double, 2> guess{0.6, 0.8};
    auto result = utils::eigen::power_method(M, guess);

    REQUIRE(result.has_value());
    CHECK(result->eigenvalue == Catch::Approx(3.0).epsilon(1e-8));
    CHECK(result->eigenvector.norm() == Catch::Approx(1.0).epsilon(1e-8));
}

TEST_CASE("power_method with bad initial guess still converges",
          "[eigen][power_method]") {
    // Diagonal matrix with dominant eigenvalue 5
    Matrix<double, 3, 3> M{
        {5.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 1.0},
    };

    // Start with a guess that has a small component along e_0.
    // The method should still converge to eigenvalue 5.
    Vector<double, 3> guess{0.01, 1.0, 1.0};
    auto result = utils::eigen::power_method(M, guess);

    REQUIRE(result.has_value());
    CHECK(result->eigenvalue == Catch::Approx(5.0).epsilon(1e-8));
    CHECK(std::abs(result->eigenvector(0)) == Catch::Approx(1.0).epsilon(1e-6));
}
