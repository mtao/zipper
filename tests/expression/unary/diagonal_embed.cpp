#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/unary/DiagonalEmbed.hpp>
#include <zipper/expression/unary/DiagonalMatrix.hpp>

#include "../../catch_include.hpp"

#include <type_traits>

TEST_CASE("diagonal_embed_static_vector", "[diagonal_embed][unary]") {
    zipper::Vector<double, 3> values{2.0, 3.0, 4.0};
    zipper::expression::unary::DiagonalEmbed diagonal(values.expression());

    CHECK(diagonal.extent(0) == 3);
    CHECK(diagonal.extent(1) == 3);
    CHECK(diagonal(0, 0) == 2.0);
    CHECK(diagonal(1, 1) == 3.0);
    CHECK(diagonal(2, 2) == 4.0);
    CHECK(diagonal(0, 2) == 0.0);
    CHECK(diagonal(2, 1) == 0.0);

    zipper::Matrix<double, 3, 3> materialized(diagonal);
    CHECK(materialized(1, 1) == 3.0);
    CHECK(materialized(1, 2) == 0.0);
}

TEST_CASE("diagonal_embed_is_lazy", "[diagonal_embed][unary][view]") {
    zipper::Vector<double, 2> values{1.0, 2.0};
    zipper::expression::unary::DiagonalEmbed diagonal(values.expression());

    values(1) = 7.0;

    CHECK(diagonal(1, 1) == 7.0);
    CHECK(diagonal(0, 1) == 0.0);
}

TEST_CASE("diagonal_embed_dynamic_vector", "[diagonal_embed][unary][dynamic]") {
    zipper::VectorX<double> values(2);
    values(0) = 5.0;
    values(1) = 8.0;
    zipper::expression::unary::DiagonalEmbed diagonal(values.expression());

    CHECK(diagonal.extent(0) == 2);
    CHECK(diagonal.extent(1) == 2);
    CHECK(diagonal(0, 0) == 5.0);
    CHECK(diagonal(0, 1) == 0.0);
    CHECK(diagonal(1, 1) == 8.0);
}

TEST_CASE("diagonal_embed_reports_diagonal_index_sets",
          "[diagonal_embed][unary][index_set]") {
    zipper::Vector<double, 3> values{2.0, 3.0, 4.0};
    zipper::expression::unary::DiagonalEmbed diagonal(values.expression());

    CHECK(*diagonal.col_range_for_row(2).begin() == 2);
    CHECK(*diagonal.row_range_for_col(1).begin() == 1);
}

TEST_CASE("diagonal_matrix_alias_matches_diagonal_embed",
          "[diagonal_matrix][diagonal_embed][unary]") {
    zipper::Vector<double, 2> values{6.0, 9.0};
    using expression_type = const decltype(values.expression()) &;
    using diagonal_type =
        zipper::expression::unary::DiagonalEmbed<expression_type>;
    using compatibility_type =
        zipper::expression::unary::DiagonalMatrix<expression_type>;
    STATIC_CHECK(std::is_same_v<compatibility_type, diagonal_type>);

    compatibility_type diagonal(values.expression());
    CHECK(diagonal(0, 0) == 6.0);
    CHECK(diagonal(0, 1) == 0.0);
    CHECK(diagonal(1, 1) == 9.0);
}
