#include <iostream>
#include <zipper/Matrix.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/ArrayBase.hxx>

#include "../../catch_include.hpp"

TEST_CASE("test_vector_scalar", "[vector][unary][scalar_arithmetic]") {
    zipper::Vector<double, 3> x{0.5, 1.5, 2.5};

    {
        auto y = 3 * x;
        CHECK(y(0) == 1.5);
        CHECK(y(1) == 4.5);
        CHECK(y(2) == 7.5);
    }
    {
        // Capture as_array() separately to avoid dangling reference:
        // as_array() on a const/rvalue returns a temporary ArrayBase that
        // owns a copy; operator+ would hold a reference into that temporary.
        auto xa = x.as_array();
        auto y = xa + 1;
        CHECK(y(0) == 1.5);
        CHECK(y(1) == 2.5);
        CHECK(y(2) == 3.5);
    }
}
TEST_CASE("test_matrix_scalar", "[matrix][unary][scalar_arithmetic]") {
    zipper::Matrix<double, 1, 3> x{{0.5, 1.5, 2.5}};

    {
        auto y = 3 * x;
        CHECK(y(0, 0) == 1.5);
        CHECK(y(0, 1) == 4.5);
        CHECK(y(0, 2) == 7.5);
    }
    {
        auto xa = x.as_array();
        auto y = xa + 1;
        CHECK(y(0, 0) == 1.5);
        CHECK(y(0, 1) == 2.5);
        CHECK(y(0, 2) == 3.5);
    }
}
