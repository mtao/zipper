// Tests for Diagonal expression:
// - Read access on const diagonal
// - Write-through on mutable diagonal
// - Diagonal of expression results
// - Extents correctness

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Constant.hpp>

#include "../../catch_include.hpp"

TEST_CASE("diagonal_read_access_square", "[diagonal][read]") {
    zipper::Matrix<double, 4, 4> M;
    for (zipper::index_type i = 0; i < 4; ++i) {
        for (zipper::index_type j = 0; j < 4; ++j) {
            M(i, j) = static_cast<double>(i * 10 + j);
        }
    }

    auto diag = M.diagonal();
    REQUIRE(diag.extents().rank() == 1);
    REQUIRE(diag.extent(0) == 4);

    CHECK(diag(0) == M(0, 0));
    CHECK(diag(1) == M(1, 1));
    CHECK(diag(2) == M(2, 2));
    CHECK(diag(3) == M(3, 3));
}

TEST_CASE("diagonal_read_access_const", "[diagonal][read]") {
    const zipper::Matrix<double, 3, 3> M{
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    auto diag = M.diagonal();
    CHECK(diag(0) == 1.0);
    CHECK(diag(1) == 5.0);
    CHECK(diag(2) == 9.0);
}

TEST_CASE("diagonal_write_through", "[diagonal][write]") {
    zipper::Matrix<double, 3, 3> M{
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    auto diag = M.diagonal();
    diag(0) = 100.0;
    diag(1) = 200.0;
    diag(2) = 300.0;

    // Changes should be reflected in the matrix
    CHECK(M(0, 0) == 100.0);
    CHECK(M(1, 1) == 200.0);
    CHECK(M(2, 2) == 300.0);

    // Off-diagonal should be unchanged
    CHECK(M(0, 1) == 2.0);
    CHECK(M(0, 2) == 3.0);
    CHECK(M(1, 0) == 4.0);
    CHECK(M(1, 2) == 6.0);
    CHECK(M(2, 0) == 7.0);
    CHECK(M(2, 1) == 8.0);
}

TEST_CASE("diagonal_assign_constant", "[diagonal][write]") {
    zipper::Matrix<double, 3, 3> M{
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    M.diagonal() =
        zipper::expression::nullary::Constant<double, 3>(0.0);

    CHECK(M(0, 0) == 0.0);
    CHECK(M(1, 1) == 0.0);
    CHECK(M(2, 2) == 0.0);
    CHECK(M(0, 1) == 2.0);
    CHECK(M(1, 0) == 4.0);
}

TEST_CASE("diagonal_assign_vector", "[diagonal][write]") {
    zipper::Matrix<double, 3, 3> M{
        {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    zipper::Vector<double, 3> v{10.0, 20.0, 30.0};

    M.diagonal() = v;
    CHECK(M(0, 0) == 10.0);
    CHECK(M(1, 1) == 20.0);
    CHECK(M(2, 2) == 30.0);
    CHECK(M(0, 1) == 0.0);
}

TEST_CASE("diagonal_reflects_matrix_mutations", "[diagonal][view]") {
    zipper::Matrix<double, 3, 3> M{
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    auto diag = M.diagonal();
    CHECK(diag(1) == 5.0);

    M(1, 1) = 42.0;
    CHECK(diag(1) == 42.0);

    M(2, 2) = 77.0;
    CHECK(diag(2) == 77.0);
}

TEST_CASE("diagonal_eval_materializes", "[diagonal][eval]") {
    zipper::Matrix<double, 3, 3> M{
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    auto d = M.diagonal().eval();
    M(0, 0) = 999.0;
    CHECK(d(0) == 1.0);  // eval'd, so independent
}

TEST_CASE("diagonal_used_in_trace", "[diagonal][trace]") {
    zipper::Matrix<double, 3, 3> M{
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

    CHECK(M.trace() == 1.0 + 5.0 + 9.0);

    M.diagonal() =
        zipper::expression::nullary::Constant<double, 3>(1.0);
    CHECK(M.trace() == 3.0);
}

TEST_CASE("diagonal_dynamic_matrix", "[diagonal][dynamic]") {
    zipper::Matrix<double, std::dynamic_extent, std::dynamic_extent> M(3, 3);
    for (zipper::index_type i = 0; i < 3; ++i) {
        for (zipper::index_type j = 0; j < 3; ++j) {
            M(i, j) = static_cast<double>(i * 10 + j);
        }
    }

    auto diag = M.diagonal();
    CHECK(diag(0) == 0.0);
    CHECK(diag(1) == 11.0);
    CHECK(diag(2) == 22.0);

    diag(0) = 99.0;
    CHECK(M(0, 0) == 99.0);
}
