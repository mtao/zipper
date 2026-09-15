
#include "../../catch_include.hpp"
#include <zipper/CSMatrix.hpp>
#include <zipper/CSRMatrix.hpp>
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

    using From = decltype(result)::expression_type;
    using To = decltype(A)::expression_type;
    using expression::detail::HasAssignTo;
    STATIC_CHECK(HasAssignTo<From, To>);
    STATIC_CHECK(HasAssignTo<From, MatrixXX<double>::expression_type>);
    STATIC_CHECK(HasAssignTo<From, decltype(A)::span_type::expression_type>);
    STATIC_CHECK_FALSE(HasAssignTo<From, const To>);
    STATIC_CHECK_FALSE(HasAssignTo<From, decltype(A)::const_span_type::expression_type>);
    STATIC_CHECK_FALSE(HasAssignTo<From, From>);
    STATIC_CHECK_FALSE(HasAssignTo<From, Matrix<double, 4, 3>::expression_type>);
    STATIC_CHECK_FALSE(HasAssignTo<From, Vector<double, 3>::expression_type>);

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

TEST_CASE("partial_transform_transpose_alias",
          "[expression][unary][partial_transform][assignment]") {
    Matrix<double, 3, 3> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    const auto expected = (2.0 * A.transpose()).eval();
    int calls = 0;
    auto transposed = A.transpose();
    A = transposed.rowwise().transform([&](const auto &row) {
        ++calls;
        return 2.0 * row;
    });
    CHECK(A == expected);
    CHECK(calls == 3);
}

TEST_CASE("partial_transform_overlapping_fibers",
          "[expression][unary][partial_transform][assignment]") {
    Matrix<double, 2, 4> A{{1, 2, 3, 4}, {5, 6, 7, 8}};
    auto source = A.leftCols<3>();
    auto target = A.rightCols<3>();
    int calls = 0;
    target = source.colwise().transform([&](const auto &col) {
        ++calls;
        return 2.0 * col;
    });
    CHECK(A == Matrix<double, 2, 4>{{1, 2, 4, 6}, {5, 10, 12, 14}});
    CHECK(calls == 3);
}

TEST_CASE("partial_transform_lazy_callback_reads_original_fiber",
          "[expression][unary][partial_transform][assignment]") {
    Matrix<double, 2, 3> A{{1, 2, 3}, {4, 5, 6}};
    int calls = 0;
    int coefficients = 0;
    A = A.rowwise().transform([&](const auto &row) {
        ++calls;
        // The reduction is deferred until each result coefficient is read.
        return [&row, &coefficients](index_type) {
            ++coefficients;
            return row.as_array().sum();
        };
    });
    CHECK(A == Matrix<double, 2, 3>{{6, 6, 6}, {15, 15, 15}});
    CHECK(calls == 2);
    CHECK(coefficients == 6);
}

TEST_CASE("partial_transform_resize_alias",
          "[expression][unary][partial_transform][assignment]") {
    MatrixXX<double> A = Matrix<double, 2, 3>{{1, 2, 3}, {4, 5, 6}};
    int calls = 0;
    SECTION("transpose_changes_mapping_before_evaluation") {
        auto transposed = A.transpose();
        A = transposed.colwise().transform([&](const auto &col) {
            ++calls;
            return 2.0 * col;
        });
        CHECK(A == Matrix<double, 3, 2>{{2, 8}, {4, 10}, {6, 12}});
        CHECK(calls == 2);
    }
    SECTION("shrinking_invalidates_source_tail") {
        auto tail = A.rightCols(2);
        A = tail.rowwise().transform([&](const auto &row) {
            ++calls;
            return 2.0 * row;
        });
        CHECK(A == Matrix<double, 2, 2>{{4, 6}, {10, 12}});
        CHECK(calls == 2);
    }
    SECTION("growing_invalidates_callback_capture") {
        const Matrix<double, 3, 4> source{{1, 2, 3, 4},
                                          {5, 6, 7, 8},
                                          {9, 10, 11, 12}};
        A = source.rowwise().transform([&](const auto &row) {
            ++calls;
            CHECK(A.rows() == 2);
            CHECK(A.cols() == 3);
            return row * A.as_array().sum();
        });
        CHECK(A == (source * 21.0).eval());
        CHECK(calls == 3);
    }
}

TEST_CASE("partial_transform_assignment_evaluates_once_per_fiber",
          "[expression][unary][partial_transform][assignment]") {
    const Matrix<double, 2, 3> source{{1, 2, 3}, {4, 5, 6}};
    int calls = 0;
    auto transformed = source.colwise().transform([&](const auto &col) {
        ++calls;
        return col * col.as_array().sum();
    });
    STATIC_CHECK_FALSE(decltype(transformed)::expression_type::traits::
                           is_coefficient_consistent);
    const Matrix<double, 2, 3> expected{{5, 14, 27}, {20, 35, 54}};
    SECTION("assignment_to_independent_dynamic_target") {
        MatrixXX<double> target(1, 1);
        target = transformed;
        CHECK(target == expected);
    }
    SECTION("eval") {
        CHECK(transformed.eval() == expected);
    }
    CHECK(calls == 3);
}

TEST_CASE("partial_transform_sparse_child_does_not_imply_sparse_output",
          "[expression][unary][partial_transform][assignment][sparse]") {
    COOMatrix<double, 2, 3> coo;
    coo.emplace(0, 1) = 2.0;
    coo.compress();
    const auto source = coo.to_csr();
    int calls = 0;
    auto transformed = source.rowwise().transform([&](const auto &row) {
        ++calls;
        return row.as_array() + 1.0;
    });
    using traits = typename decltype(transformed)::expression_type::traits;
    STATIC_CHECK_FALSE(traits::has_index_set);
    STATIC_CHECK_FALSE(detail::is_sparse_layout_preference_v<
                       typename traits::preferred_layout>);
    auto result = transformed.eval();
    STATIC_CHECK(std::is_same_v<decltype(result), Matrix<double, 2, 3>>);
    CHECK(result == Matrix<double, 2, 3>{{1, 3, 1}, {1, 1, 1}});
    CHECK(calls == 2);
}
