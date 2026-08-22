#include "../catch_include.hpp"

#include <zipper/Array.hpp>
#include <zipper/Matrix.hpp>

TEST_CASE("public_matrix_expressions_work_together", "[integration][matrix]") {
    zipper::Matrix<double, 2, 2> a{{1.0, 2.0}, {3.0, 4.0}};
    zipper::Matrix<double, 2, 2> b{{5.0, 6.0}, {7.0, 8.0}};

    auto scaled = (2.0 * a).eval();
    auto sum = (a + b).eval();
    auto coefficient_product =
        (a.as_array() * b.as_array()).eval();
    auto product = (a * b).eval();
    auto cast = a.cast<int>().eval();

    CHECK(scaled(1, 0) == 6.0);
    CHECK(sum(0, 1) == 8.0);
    CHECK(coefficient_product(1, 1) == 32.0);
    CHECK(product(0, 0) == 19.0);
    CHECK(product(1, 1) == 50.0);
    CHECK(cast(0, 1) == 2);

    b.as_array() = coefficient_product;
    CHECK(b(0, 0) == 5.0);
    CHECK(b(1, 1) == 32.0);
}
