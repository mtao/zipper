
#include <zipper/utils/format.hpp>

#include <zipper/Array.hpp>
#include <zipper/Container.hpp>
#include <zipper/Form.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include "../catch_include.hpp"

// ---- Vector (rank 1, has begin/end) ----
TEST_CASE("format_vector", "[format]") {
    zipper::Vector<double, 3> x({1, 2, 0});
    CHECK(fmt::format("{}", x) == "[1, 2, 0]");
}

// ---- VectorBase view (rank 1, no begin/end) ----
TEST_CASE("format_vector_head", "[format]") {
    zipper::Vector<double, 3> x({1, 2, 0});
    CHECK(fmt::format("{}", x.head<2>()) == "[1, 2]");
}

// ---- Matrix (rank 2) ----
TEST_CASE("format_matrix", "[format]") {
    zipper::Matrix<double, 3, 3> x;
    x.col(0) = {0, 1, 2};
    x.col(1) = {-3, 3, 0};
    x.col(2) = {100, 4, 1};

    CHECK(fmt::format("{}", x) == "[[0, -3, 100];[1, 3, 4];[2, 0, 1]]");
}

// ---- MatrixBase view (transpose) ----
TEST_CASE("format_matrix_transpose", "[format]") {
    zipper::Matrix<double, 2, 3> M;
    M(0, 0) = 1; M(0, 1) = 2; M(0, 2) = 3;
    M(1, 0) = 4; M(1, 1) = 5; M(1, 2) = 6;

    CHECK(fmt::format("{}", M.transpose()) == "[[1, 4];[2, 5];[3, 6]]");
}

// ---- Array rank 1 ----
TEST_CASE("format_array_1d", "[format]") {
    zipper::Array<double, 3> a;
    a(0) = 10; a(1) = 20; a(2) = 30;
    CHECK(fmt::format("{}", a) == "[10, 20, 30]");
}

// ---- Array rank 2 ----
TEST_CASE("format_array_2d", "[format]") {
    zipper::Array<double, 2, 3> a;
    a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
    a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;

    CHECK(fmt::format("{}", a) == "[[1, 2, 3];[4, 5, 6]]");
}

// ---- Container (rank 1, has begin/end) ----
TEST_CASE("format_container", "[format]") {
    zipper::Container<double, 3> c;
    c(0) = 7; c(1) = 8; c(2) = 9;
    CHECK(fmt::format("{}", c) == "[7, 8, 9]");
}

// ---- Form (rank 1) ----
TEST_CASE("format_form", "[format]") {
    zipper::Form<double, 3> f;
    f(0) = 100; f(1) = 200; f(2) = 300;
    CHECK(fmt::format("{}", f) == "[100, 200, 300]");
}

// ---- Tensor rank 1 ----
TEST_CASE("format_tensor_1d", "[format]") {
    zipper::Tensor<double, 3> t;
    t(0) = 11; t(1) = 22; t(2) = 33;
    CHECK(fmt::format("{}", t) == "[11, 22, 33]");
}

// ---- Tensor rank 2 ----
TEST_CASE("format_tensor_2d", "[format]") {
    zipper::Tensor<double, 2, 3> t;
    t(0, 0) = 1; t(0, 1) = 2; t(0, 2) = 3;
    t(1, 0) = 4; t(1, 1) = 5; t(1, 2) = 6;

    CHECK(fmt::format("{}", t) == "[[1, 2, 3];[4, 5, 6]]");
}

// ---- Tensor rank 3 (multi-level recursion) ----
TEST_CASE("format_tensor_3d", "[format]") {
    zipper::Tensor<double, 2, 3, 4> t;
    for (zipper::index_type i = 0; i < 2; ++i)
        for (zipper::index_type j = 0; j < 3; ++j)
            for (zipper::index_type k = 0; k < 4; ++k)
                t(i, j, k) = static_cast<double>(100 * i + 10 * j + k);

    CHECK(fmt::format("{}", t) ==
          "[[[0, 1, 2, 3];[10, 11, 12, 13];[20, 21, 22, 23]];"
          "[[100, 101, 102, 103];[110, 111, 112, 113];[120, 121, 122, 123]]]");
}
