// Compile-check for the code examples in
// include/zipper/expression/unary/mappings.md, with their documented results.

#include <zipper/FormBase.hxx>
#include <zipper/Matrix.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/MatrixBase.hpp>
#include <zipper/concepts/Matrix.hpp>

#include <type_traits>

#include "catch_include.hpp"

TEST_CASE("mappings_meshgrid", "[mappings]") {
    zipper::Vector<double, 3> x{0.0, 0.5, 1.0};
    zipper::Vector<double, 4> y{0.0, 1.0, 2.0, 3.0};

    // X(i, j) = x(i), Y(i, j) = y(j): like NumPy's meshgrid(x, y, indexing="ij")
    zipper::Matrix X = x.broadcast<1>(zipper::extents<3, 4>{});
    zipper::Matrix Y = y.broadcast<0>(zipper::extents<3, 4>{});

    STATIC_CHECK(std::is_same_v<decltype(X), zipper::Matrix<double, 3, 4>>);
    CHECK(X(2, 3) == 1.0);
    CHECK(Y(2, 3) == 3.0);
    CHECK(Y(0, 1) == 1.0);
}

TEST_CASE("mappings_colwise_norm", "[mappings]") {
    zipper::Matrix<double, 3, 100> A;
    for (zipper::index_type j = 0; j < 100; ++j) {
        A(0, j) = 3.0;
        A(1, j) = 4.0;
        A(2, j) = 0.0;
    }
    zipper::Vector x = A.colwise().norm();
    STATIC_CHECK(std::is_same_v<decltype(x), zipper::Vector<double, 100>>);
    CHECK(x(42) == 5.0);

    // colwise() is over<0>(): the same reduction over axis 0.
    zipper::Vector y = A.over<0>().norm();
    CHECK(y(42) == 5.0);
}

TEST_CASE("mappings_tensor_diagonals", "[mappings]") {
    zipper::Tensor<double, 40, 20, 5, 5> T = {};
    T(7, 3, 2, 2) = 1.5;

    // D is a 5x40x20 tensor whose first axis is the diagonals of each 5x5 matrix
    zipper::Tensor D = T.over<2, 3>().map<0>([](const zipper::concepts::MatrixExpression auto &A) {
        return zipper::MatrixBase(A).diagonal();
    });

    STATIC_CHECK(std::is_same_v<decltype(D), zipper::Tensor<double, 5, 40, 20>>);
    CHECK(D(2, 7, 3) == 1.5);
    CHECK(D(1, 7, 3) == 0.0);
}

TEST_CASE("mappings_rowwise_forms", "[mappings]") {
    zipper::Matrix<double, 2, 3> A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    zipper::Vector<double, 3> x{1.0, 0.0, -1.0};

    // Each row read as a 1-form applied to x: b = A x
    zipper::Vector b = A.rowwise().map([&](auto const &r) {
        return zipper::FormBase(r) * x;
    });

    STATIC_CHECK(std::is_same_v<decltype(b), zipper::Vector<double, 2>>);
    CHECK(b(0) == -2.0);
    CHECK(b(1) == -2.0);
}
