
#include "../../catch_include.hpp"
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/MDArray.hpp>
#include <zipper/expression/unary/PartialTransform.hpp>

using namespace zipper;

TEST_CASE("partial_transform_colwise_identity",
          "[expression][unary][partial_transform]") {
    // colwise transform with identity function should be a no-op
    Matrix<double, 3, 4> A;
    for (index_type i = 0; i < 3; ++i)
        for (index_type j = 0; j < 4; ++j)
            A(i, j) = static_cast<double>(i * 4 + j + 1);

    // colwise: Indices={0}, iterates columns, fn receives each column
    // Note: fn must accept by const reference (fibers are non-copyable views)
    auto result = A.colwise().transform([](const auto &col) {
        return col.eval(); // materialize to return owned data
    });

    // Check that result has same shape
    REQUIRE(result.extent(0) == 3);
    REQUIRE(result.extent(1) == 4);

    // Check values are unchanged
    for (index_type i = 0; i < 3; ++i)
        for (index_type j = 0; j < 4; ++j) CHECK(result(i, j) == A(i, j));
}

TEST_CASE("partial_transform_rowwise_identity",
          "[expression][unary][partial_transform]") {
    Matrix<double, 3, 4> A;
    for (index_type i = 0; i < 3; ++i)
        for (index_type j = 0; j < 4; ++j)
            A(i, j) = static_cast<double>(i * 4 + j + 1);

    // rowwise: Indices={1}, iterates rows, fn receives each row
    auto result =
        A.rowwise().transform([](const auto &row) { return row.eval(); });

    REQUIRE(result.extent(0) == 3);
    REQUIRE(result.extent(1) == 4);

    for (index_type i = 0; i < 3; ++i)
        for (index_type j = 0; j < 4; ++j) CHECK(result(i, j) == A(i, j));
}

TEST_CASE("partial_transform_colwise_scale",
          "[expression][unary][partial_transform]") {
    // Scale each column by 2
    Matrix<double, 3, 4> A;
    for (index_type i = 0; i < 3; ++i)
        for (index_type j = 0; j < 4; ++j)
            A(i, j) = static_cast<double>(i * 4 + j + 1);

    Matrix<double, 3, 4> B;
    B = A.colwise().transform(
        [](const auto &col) { return (2.0 * col).eval(); });

    for (index_type i = 0; i < 3; ++i)
        for (index_type j = 0; j < 4; ++j) CHECK(B(i, j) == 2.0 * A(i, j));
}

TEST_CASE("partial_transform_rowwise_scale",
          "[expression][unary][partial_transform]") {
    // Scale each row by 3
    Matrix<double, 3, 4> A;
    for (index_type i = 0; i < 3; ++i)
        for (index_type j = 0; j < 4; ++j)
            A(i, j) = static_cast<double>(i * 4 + j + 1);

    Matrix<double, 3, 4> B;
    B = A.rowwise().transform(
        [](const auto &row) { return (3.0 * row).eval(); });

    for (index_type i = 0; i < 3; ++i)
        for (index_type j = 0; j < 4; ++j) CHECK(B(i, j) == 3.0 * A(i, j));
}

TEST_CASE("partial_transform_colwise_normalize",
          "[expression][unary][partial_transform]") {
    // Normalize each column
    Matrix<double, 2, 3> A;
    A(0, 0) = 3.0;
    A(1, 0) = 4.0; // col 0: norm = 5
    A(0, 1) = 1.0;
    A(1, 1) = 0.0; // col 1: norm = 1
    A(0, 2) = 0.0;
    A(1, 2) = 2.0; // col 2: norm = 2

    Matrix<double, 2, 3> B;
    B = A.colwise().transform(
        [](const auto &col) { return col.normalized().eval(); });

    CHECK(B(0, 0) == Catch::Approx(3.0 / 5.0));
    CHECK(B(1, 0) == Catch::Approx(4.0 / 5.0));
    CHECK(B(0, 1) == Catch::Approx(1.0));
    CHECK(B(1, 1) == Catch::Approx(0.0));
    CHECK(B(0, 2) == Catch::Approx(0.0));
    CHECK(B(1, 2) == Catch::Approx(1.0));
}

TEST_CASE("partial_transform_givens_rotation",
          "[expression][unary][partial_transform]") {
    // Apply a Givens-like rotation to each column: G * col
    // This demonstrates the core use case for SVD Givens rotations.

    // Create a 2x3 matrix
    Matrix<double, 2, 3> A;
    A(0, 0) = 1.0;
    A(0, 1) = 0.0;
    A(0, 2) = 3.0;
    A(1, 0) = 0.0;
    A(1, 1) = 1.0;
    A(1, 2) = 4.0;

    // 2x2 rotation by 45 degrees
    double c = std::cos(M_PI / 4.0);
    double s = std::sin(M_PI / 4.0);
    Matrix<double, 2, 2> G;
    G(0, 0) = c;
    G(0, 1) = -s;
    G(1, 0) = s;
    G(1, 1) = c;

    // Apply G to each column: result_col = G * col
    Matrix<double, 2, 3> B;
    B = A.colwise().transform(
        [&G](const auto &col) { return (G * col).eval(); });

    // Col 0: G * [1, 0]^T = [c, s]^T
    CHECK(B(0, 0) == Catch::Approx(c));
    CHECK(B(1, 0) == Catch::Approx(s));
    // Col 1: G * [0, 1]^T = [-s, c]^T
    CHECK(B(0, 1) == Catch::Approx(-s));
    CHECK(B(1, 1) == Catch::Approx(c));
    // Col 2: G * [3, 4]^T = [3c - 4s, 3s + 4c]^T
    CHECK(B(0, 2) == Catch::Approx(3.0 * c - 4.0 * s));
    CHECK(B(1, 2) == Catch::Approx(3.0 * s + 4.0 * c));
}

TEST_CASE("partial_transform_assign_to_dynamic",
          "[expression][unary][partial_transform]") {
    // Test with dynamic-extent matrices
    MatrixXX<double> A(3, 4);
    for (index_type i = 0; i < 3; ++i)
        for (index_type j = 0; j < 4; ++j)
            A(i, j) = static_cast<double>(i * 4 + j + 1);

    MatrixXX<double> B(3, 4);
    B = A.colwise().transform(
        [](const auto &col) { return (2.0 * col).eval(); });

    for (index_type i = 0; i < 3; ++i)
        for (index_type j = 0; j < 4; ++j) CHECK(B(i, j) == 2.0 * A(i, j));
}

TEST_CASE("partial_transform_coeff_access",
          "[expression][unary][partial_transform]") {
    // Test the slow per-element coeff() path (used in sub-expressions)
    Matrix<double, 2, 3> A;
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(0, 2) = 3.0;
    A(1, 0) = 4.0;
    A(1, 1) = 5.0;
    A(1, 2) = 6.0;

    auto result = A.colwise().transform(
        [](const auto &col) { return (10.0 * col).eval(); });

    // Accessing via coeff() should give the correct values
    CHECK(result(0, 0) == 10.0);
    CHECK(result(0, 1) == 20.0);
    CHECK(result(0, 2) == 30.0);
    CHECK(result(1, 0) == 40.0);
    CHECK(result(1, 1) == 50.0);
    CHECK(result(1, 2) == 60.0);
}
