#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/orthogonalization/gram_schmidt.hpp>

#include "catch_include.hpp"

using namespace zipper;

namespace {

template <concepts::Matrix MatrixType>
void check_orthonormal_columns(const MatrixType &matrix) {
    for (index_type i = 0; i < matrix.extent(1); ++i) {
        const VectorX<double> column_i(matrix.col(i));
        CHECK(column_i.norm<2>() == Catch::Approx(1.0).margin(1e-12));

        for (index_type j = 0; j < i; ++j) {
            const VectorX<double> column_j(matrix.col(j));
            CHECK(column_i.dot(column_j) ==
                  Catch::Approx(0.0).margin(1e-12));
        }
    }
}

} // namespace

TEST_CASE("gram_schmidt_returns_orthonormal_copy",
          "[utils][orthogonalization][gram_schmidt]") {
    const Matrix<double, 3, 2> input{
        {1.0, 1.0},
        {1.0, 0.0},
        {0.0, 1.0},
    };

    const auto output = utils::orthogonalization::gram_schmidt(input);

    CHECK(input(0, 0) == 1.0);
    CHECK(input(1, 0) == 1.0);
    CHECK(input(2, 0) == 0.0);
    CHECK(input(0, 1) == 1.0);
    CHECK(input(1, 1) == 0.0);
    CHECK(input(2, 1) == 1.0);
    check_orthonormal_columns(output);
}

TEST_CASE("gram_schmidt_in_place_orthonormalizes_columns",
          "[utils][orthogonalization][gram_schmidt]") {
    Matrix<double, 3, 2> matrix{
        {2.0, 1.0},
        {0.0, 1.0},
        {0.0, 1.0},
    };

    utils::orthogonalization::gram_schmidt_in_place(matrix);

    CHECK(matrix(0, 0) == Catch::Approx(1.0));
    CHECK(matrix(1, 0) == Catch::Approx(0.0).margin(1e-12));
    CHECK(matrix(2, 0) == Catch::Approx(0.0).margin(1e-12));
    check_orthonormal_columns(matrix);
}

TEST_CASE("gram_schmidt_zeros_dependent_columns",
          "[utils][orthogonalization][gram_schmidt]") {
    Matrix<double, 3, 3> matrix{
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0},
        {0.0, 0.0, 0.0},
    };

    utils::orthogonalization::gram_schmidt_in_place(matrix);

    CHECK(Vector<double, 3>(matrix.col(0)).norm<2>() ==
          Catch::Approx(1.0));
    CHECK(Vector<double, 3>(matrix.col(1)).norm<2>() ==
          Catch::Approx(1.0));
    CHECK(Vector<double, 3>(matrix.col(0)).dot(
              Vector<double, 3>(matrix.col(1))) ==
          Catch::Approx(0.0).margin(1e-12));
    CHECK(Vector<double, 3>(matrix.col(2)).norm<2>() ==
          Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("gram_schmidt_supports_dynamic_extents",
          "[utils][orthogonalization][gram_schmidt][dynamic]") {
    MatrixXX<double> matrix(4, 3);
    matrix = Matrix<double, 4, 3>{
        {1.0, 1.0, 0.0},
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0},
        {1.0, 2.0, 3.0},
    };

    const auto output = utils::orthogonalization::gram_schmidt(matrix);

    CHECK(output.extent(0) == 4);
    CHECK(output.extent(1) == 3);
    check_orthonormal_columns(output);

    utils::orthogonalization::gram_schmidt_in_place(matrix);
    check_orthonormal_columns(matrix);
}
