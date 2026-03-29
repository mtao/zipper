// Tests for operations on dynamic-extent vectors/matrices.
// Mirrors static-extent tests from various files to ensure dynamic-extent
// code paths work correctly.

#include "../catch_include.hpp"
#include <zipper/ArrayBase.hxx>
#include <zipper/Form.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/MatrixBase.hxx>
#include <zipper/Vector.hpp>
#include <zipper/expression/binary/CrossProduct.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/MDArray.hpp>
#include <zipper/expression/nullary/Unit.hpp>
#include <zipper/expression/reductions/CoefficientSum.hpp>
#include <zipper/expression/reductions/Determinant.hpp>
#include <zipper/expression/reductions/LpNorm.hpp>
#include <zipper/expression/reductions/Trace.hpp>
#include <zipper/expression/unary/Cast.hpp>
#include <zipper/expression/unary/Cofactor.hpp>
#include <zipper/expression/unary/Homogeneous.hpp>
#include <zipper/expression/unary/PartialReduction.hpp>
#include <zipper/expression/unary/Reshape.hpp>
#include <zipper/expression/unary/detail/PartialReductionDispatcher.hpp>
#include <zipper/utils/determinant.hpp>

using namespace zipper;

// ============================================================
// Binary operations (element-wise via ArrayBase)
// ============================================================

TEST_CASE("dynamic_binary_vector_add", "[dynamic][binary]") {
    VectorX<double> a = {1.0, 2.0, 3.0};
    VectorX<double> b = {4.0, 5.0, 6.0};

    auto c = a.as_array() + b.as_array();
    REQUIRE(c.extent(0) == 3);
    CHECK(c(0) == 5.0);
    CHECK(c(1) == 7.0);
    CHECK(c(2) == 9.0);
}

TEST_CASE("dynamic_binary_vector_subtract", "[dynamic][binary]") {
    VectorX<double> a = {10.0, 20.0, 30.0};
    VectorX<double> b = {1.0, 2.0, 3.0};

    auto c = a.as_array() - b.as_array();
    REQUIRE(c.extent(0) == 3);
    CHECK(c(0) == 9.0);
    CHECK(c(1) == 18.0);
    CHECK(c(2) == 27.0);
}

TEST_CASE("dynamic_binary_matrix_multiply", "[dynamic][binary]") {
    MatrixXX<double> A(2, 2);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;

    MatrixXX<double> B(2, 2);
    B(0, 0) = 5;
    B(0, 1) = 6;
    B(1, 0) = 7;
    B(1, 1) = 8;

    auto C = A.as_array() * B.as_array();
    REQUIRE(C.extent(0) == 2);
    REQUIRE(C.extent(1) == 2);
    CHECK(C(0, 0) == 5.0);
    CHECK(C(0, 1) == 12.0);
    CHECK(C(1, 0) == 21.0);
    CHECK(C(1, 1) == 32.0);
}

TEST_CASE("dynamic_binary_matrix_divide", "[dynamic][binary]") {
    MatrixXX<double> A(2, 2);
    A(0, 0) = 10;
    A(0, 1) = 20;
    A(1, 0) = 30;
    A(1, 1) = 40;

    MatrixXX<double> B(2, 2);
    B(0, 0) = 2;
    B(0, 1) = 4;
    B(1, 0) = 5;
    B(1, 1) = 8;

    auto C = A.as_array() / B.as_array();
    CHECK(C(0, 0) == 5.0);
    CHECK(C(0, 1) == 5.0);
    CHECK(C(1, 0) == 6.0);
    CHECK(C(1, 1) == 5.0);
}

// ============================================================
// Matrix product
// ============================================================

TEST_CASE("dynamic_matrix_product", "[dynamic][product]") {
    MatrixXX<double> A(2, 3);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;

    MatrixXX<double> B(3, 2);
    B(0, 0) = 7;
    B(0, 1) = 8;
    B(1, 0) = 9;
    B(1, 1) = 10;
    B(2, 0) = 11;
    B(2, 1) = 12;

    MatrixXX<double> C = A * B;
    REQUIRE(C.rows() == 2);
    REQUIRE(C.cols() == 2);
    // Row 0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
    CHECK(C(0, 0) == 58.0);
    CHECK(C(0, 1) == 64.0);
    // Row 1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
    CHECK(C(1, 0) == 139.0);
    CHECK(C(1, 1) == 154.0);
}

TEST_CASE("dynamic_matrix_vector_product", "[dynamic][product]") {
    MatrixXX<double> A(2, 3);
    A(0, 0) = 1;
    A(0, 1) = 0;
    A(0, 2) = 0;
    A(1, 0) = 0;
    A(1, 1) = 2;
    A(1, 2) = 0;

    VectorX<double> x = {3.0, 4.0, 5.0};

    VectorX<double> y = A * x;
    REQUIRE(y.size() == 2);
    CHECK(y(0) == 3.0);
    CHECK(y(1) == 8.0);
}

// ============================================================
// Scalar arithmetic
// ============================================================

TEST_CASE("dynamic_scalar_multiply_vector", "[dynamic][scalar]") {
    VectorX<double> x = {0.5, 1.5, 2.5};

    auto y = 3.0 * x;
    REQUIRE(y.size() == 3);
    CHECK(y(0) == 1.5);
    CHECK(y(1) == 4.5);
    CHECK(y(2) == 7.5);
}

TEST_CASE("dynamic_scalar_add_array", "[dynamic][scalar]") {
    VectorX<double> x = {0.5, 1.5, 2.5};

    auto y = x.as_array() + 1.0;
    CHECK(y(0) == 1.5);
    CHECK(y(1) == 2.5);
    CHECK(y(2) == 3.5);
}

TEST_CASE("dynamic_scalar_multiply_matrix", "[dynamic][scalar]") {
    MatrixXX<double> M(2, 2);
    M(0, 0) = 1;
    M(0, 1) = 2;
    M(1, 0) = 3;
    M(1, 1) = 4;

    auto N = 2.0 * M;
    CHECK(N(0, 0) == 2.0);
    CHECK(N(0, 1) == 4.0);
    CHECK(N(1, 0) == 6.0);
    CHECK(N(1, 1) == 8.0);
}

// ============================================================
// Cross product
// ============================================================

TEST_CASE("dynamic_cross_product", "[dynamic][product]") {
    VectorX<double> a = {1.0, 0.0, 0.0};
    VectorX<double> b = {0.0, 1.0, 0.0};

    auto c = a.cross(b);
    REQUIRE(c.size() == 3);
    CHECK(c(0) == 0.0);
    CHECK(c(1) == 0.0);
    CHECK(c(2) == 1.0);
}

// ============================================================
// Partial reductions
// ============================================================

namespace {
template <typename ExpressionType, int index>
auto make_partial_sum(const ExpressionType &a,
                      std::integral_constant<int, index>) {
    return expression::unary::detail::
        PartialReductionDispatcher<const ExpressionType, index>(a)
            .sum();
}
} // namespace

TEST_CASE("dynamic_partial_sum_rowwise", "[dynamic][reduction]") {
    MatrixXX<double> A(3, 4);
    A(0, 0) = 1;
    A(0, 1) = 0;
    A(0, 2) = 0;
    A(0, 3) = 0;
    A(1, 0) = 0;
    A(1, 1) = 1;
    A(1, 2) = 0;
    A(1, 3) = 0;
    A(2, 0) = 0;
    A(2, 1) = 0;
    A(2, 2) = 1;
    A(2, 3) = 0;

    // Sum along axis 1 (cols) -> row sums
    auto pr =
        make_partial_sum(A.expression(), std::integral_constant<int, 1>{});
    REQUIRE(pr.extent(0) == 3);
    CHECK(pr(0) == 1.0);
    CHECK(pr(1) == 1.0);
    CHECK(pr(2) == 1.0);

    // Sum along axis 0 (rows) -> column sums
    auto pc =
        make_partial_sum(A.expression(), std::integral_constant<int, 0>{});
    REQUIRE(pc.extent(0) == 4);
    CHECK(pc(0) == 1.0);
    CHECK(pc(1) == 1.0);
    CHECK(pc(2) == 1.0);
    CHECK(pc(3) == 0.0);
}

// ============================================================
// Trace
// ============================================================

TEST_CASE("dynamic_trace", "[dynamic][reduction]") {
    MatrixXX<double> I = expression::nullary::
        Identity<double, std::dynamic_extent, std::dynamic_extent>(3, 3);
    CHECK(I.trace() == 3.0);

    MatrixXX<double> A(3, 3);
    A(0, 0) = 2;
    A(0, 1) = 0;
    A(0, 2) = 0;
    A(1, 0) = 0;
    A(1, 1) = 5;
    A(1, 2) = 0;
    A(2, 0) = 0;
    A(2, 1) = 0;
    A(2, 2) = 8;
    CHECK(A.trace() == 15.0);
}

// ============================================================
// Norm & dot
// ============================================================

TEST_CASE("dynamic_vector_norm", "[dynamic][reduction]") {
    VectorX<double> x = {3.0, 4.0};
    CHECK(x.norm() == Catch::Approx(5.0));
}

TEST_CASE("dynamic_vector_dot", "[dynamic][reduction]") {
    VectorX<double> a = {1.0, 2.0, 3.0};
    VectorX<double> b = {4.0, 5.0, 6.0};
    CHECK(a.dot(b) == Catch::Approx(32.0));
}

// ============================================================
// Cofactor
// ============================================================

TEST_CASE("dynamic_cofactor_2x2", "[dynamic][cofactor]") {
    MatrixXX<double> m(2, 2);
    m(0, 0) = 3;
    m(0, 1) = 8;
    m(1, 0) = 4;
    m(1, 1) = 6;

    auto c = expression::unary::Cofactor(m.expression());
    CHECK(c(0, 0) == Catch::Approx(6));
    CHECK(c(0, 1) == Catch::Approx(-4));
    CHECK(c(1, 0) == Catch::Approx(-8));
    CHECK(c(1, 1) == Catch::Approx(3));
}

TEST_CASE("dynamic_cofactor_3x3", "[dynamic][cofactor]") {
    MatrixXX<double> m(3, 3);
    m(0, 0) = 1;
    m(0, 1) = 2;
    m(0, 2) = 3;
    m(1, 0) = 0;
    m(1, 1) = 4;
    m(1, 2) = 5;
    m(2, 0) = 1;
    m(2, 1) = 0;
    m(2, 2) = 6;

    auto c = expression::unary::Cofactor(m.expression());
    CHECK(c(0, 0) == Catch::Approx(24));
    CHECK(c(0, 1) == Catch::Approx(5));
    CHECK(c(0, 2) == Catch::Approx(-4));
    CHECK(c(1, 0) == Catch::Approx(-12));
    CHECK(c(1, 1) == Catch::Approx(3));
    CHECK(c(1, 2) == Catch::Approx(2));
    CHECK(c(2, 0) == Catch::Approx(-2));
    CHECK(c(2, 1) == Catch::Approx(-5));
    CHECK(c(2, 2) == Catch::Approx(4));
}

// ============================================================
// Cast
// ============================================================

TEST_CASE("dynamic_cast_double_to_float", "[dynamic][cast]") {
    MatrixXX<double> A = expression::nullary::
        Identity<double, std::dynamic_extent, std::dynamic_extent>(3, 3);

    auto B = expression::unary::cast<float>(A.expression());
    CHECK(B(0, 0) == 1.0f);
    CHECK(B(1, 0) == 0.0f);
    CHECK(B(1, 1) == 1.0f);
    CHECK(B(2, 2) == 1.0f);
}

// ============================================================
// Reshape
// ============================================================

TEST_CASE("dynamic_reshape_vec_to_matrix", "[dynamic][reshape]") {
    VectorX<double> v = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    // Reshape 6-vector into 2x3
    using new_extents = zipper::extents<2, 3>;
    auto r = expression::unary::Reshape(v.expression(), new_extents{});

    CHECK(r(0, 0) == 1.0);
    CHECK(r(0, 1) == 2.0);
    CHECK(r(0, 2) == 3.0);
    CHECK(r(1, 0) == 4.0);
    CHECK(r(1, 1) == 5.0);
    CHECK(r(1, 2) == 6.0);
}

// ============================================================
// Mixed static/dynamic
// ============================================================

TEST_CASE("mixed_static_dynamic_add", "[dynamic][binary][mixed]") {
    Vector<double, 3> a{1.0, 2.0, 3.0};
    VectorX<double> b = {4.0, 5.0, 6.0};

    // static + dynamic
    auto c = a.as_array() + b.as_array();
    REQUIRE(c.extent(0) == 3);
    CHECK(c(0) == 5.0);
    CHECK(c(1) == 7.0);
    CHECK(c(2) == 9.0);
}

TEST_CASE("mixed_static_dynamic_product", "[dynamic][product][mixed]") {
    Matrix<double, 2, 3> A{{1, 0, 0}, {0, 2, 0}};
    VectorX<double> x = {3.0, 4.0, 5.0};

    auto y = A * x;
    REQUIRE(y.size() == 2);
    CHECK(y(0) == 3.0);
    CHECK(y(1) == 8.0);
}

// ============================================================
// Compound operations on dynamic types
// ============================================================

TEST_CASE("dynamic_compound_operations", "[dynamic][compound]") {
    VectorX<double> a = {1.0, 2.0, 3.0};
    VectorX<double> b = {4.0, 5.0, 6.0};

    // element-wise add then scalar multiply
    VectorX<double> sum = a + b;
    VectorX<double> c = sum * 2.0;
    REQUIRE(c.size() == 3);
    // (1+4)*2 = 10
    CHECK(c(0) == 10.0);
    // (2+5)*2 = 14
    CHECK(c(1) == 14.0);
    // (3+6)*2 = 18
    CHECK(c(2) == 18.0);
}

TEST_CASE("dynamic_matrix_identity_multiply", "[dynamic][product]") {
    MatrixXX<double> I = expression::nullary::
        Identity<double, std::dynamic_extent, std::dynamic_extent>(3, 3);
    MatrixXX<double> A(3, 3);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;
    A(2, 0) = 7;
    A(2, 1) = 8;
    A(2, 2) = 9;

    MatrixXX<double> B = I * A;
    for (index_type r = 0; r < 3; ++r) {
        for (index_type c = 0; c < 3; ++c) { CHECK(B(r, c) == A(r, c)); }
    }
}

// ============================================================
// Dynamic vector resize + assign
// ============================================================

TEST_CASE("dynamic_vector_resize", "[dynamic][resize]") {
    VectorX<double> v(3);
    v(0) = 1.0;
    v(1) = 2.0;
    v(2) = 3.0;
    REQUIRE(v.size() == 3);

    v.resize(5);
    REQUIRE(v.size() == 5);

    v = {10.0, 20.0, 30.0, 40.0, 50.0};
    CHECK(v(0) == 10.0);
    CHECK(v(4) == 50.0);
}

TEST_CASE("dynamic_matrix_resize", "[dynamic][resize]") {
    MatrixXX<double> M(2, 2);
    M(0, 0) = 1;
    M(0, 1) = 2;
    M(1, 0) = 3;
    M(1, 1) = 4;

    M.resize(3, 3);
    REQUIRE(M.rows() == 3);
    REQUIRE(M.cols() == 3);
}

// ============================================================
// Transpose
// ============================================================

TEST_CASE("dynamic_matrix_transpose", "[dynamic][transpose]") {
    MatrixXX<double> A(3, 3);
    for (index_type j = 0; j < 3; ++j) {
        for (index_type k = 0; k < 3; ++k) { A(j, k) = 3.0 * j + k; }
    }

    auto AT = A.transpose();
    REQUIRE(AT.extent(0) == 3);
    REQUIRE(AT.extent(1) == 3);
    for (index_type j = 0; j < 3; ++j) {
        for (index_type k = 0; k < 3; ++k) { CHECK(AT(k, j) == 3.0 * j + k); }
    }
}

TEST_CASE("dynamic_transpose_product", "[dynamic][transpose][product]") {
    MatrixXX<double> M(3, 3);
    for (index_type j = 0; j < 3; ++j) {
        for (index_type k = 0; k < 3; ++k) { M(j, k) = 3.0 * j + k; }
    }

    MatrixXX<double> MTM = M.transpose() * M;
    REQUIRE(MTM.rows() == 3);
    REQUIRE(MTM.cols() == 3);
    CHECK(MTM(0, 0) == 45.0);
    CHECK(MTM(1, 1) == 66.0);
    CHECK(MTM(2, 2) == 93.0);
}

// ============================================================
// Determinant
// ============================================================

TEST_CASE("dynamic_determinant_2x2", "[dynamic][reduction][determinant]") {
    MatrixXX<double> A(2, 2);
    A(0, 0) = 3;
    A(0, 1) = 8;
    A(1, 0) = 4;
    A(1, 1) = 6;

    double det = expression::reductions::Determinant(A.expression())();
    CHECK(det == Catch::Approx(-14.0));
}

TEST_CASE("dynamic_determinant_3x3", "[dynamic][reduction][determinant]") {
    MatrixXX<double> A(3, 3);
    A(0, 0) = 6;
    A(0, 1) = 1;
    A(0, 2) = 1;
    A(1, 0) = 4;
    A(1, 1) = -2;
    A(1, 2) = 5;
    A(2, 0) = 2;
    A(2, 1) = 8;
    A(2, 2) = 7;

    double det = zipper::utils::determinant(A);
    CHECK(det == Catch::Approx(-306.0));
}

TEST_CASE("dynamic_determinant_identity", "[dynamic][reduction][determinant]") {
    MatrixXX<double> I = expression::nullary::
        Identity<double, std::dynamic_extent, std::dynamic_extent>(4, 4);
    double det = expression::reductions::Determinant(I.expression())();
    CHECK(det == Catch::Approx(1.0));
}

// ============================================================
// Vector head / tail / segment
// ============================================================

TEST_CASE("dynamic_vector_head", "[dynamic][slicing]") {
    VectorX<double> v = {1.0, 2.0, 3.0, 4.0, 5.0};

    auto h = v.head(3);
    REQUIRE(h.size() == 3);
    CHECK(h(0) == 1.0);
    CHECK(h(1) == 2.0);
    CHECK(h(2) == 3.0);
}

TEST_CASE("dynamic_vector_tail", "[dynamic][slicing]") {
    VectorX<double> v = {1.0, 2.0, 3.0, 4.0, 5.0};

    auto t = v.tail(2);
    REQUIRE(t.size() == 2);
    CHECK(t(0) == 4.0);
    CHECK(t(1) == 5.0);
}

TEST_CASE("dynamic_vector_segment", "[dynamic][slicing]") {
    VectorX<double> v = {10.0, 20.0, 30.0, 40.0, 50.0};

    auto s = v.segment(1, 3);
    REQUIRE(s.size() == 3);
    CHECK(s(0) == 20.0);
    CHECK(s(1) == 30.0);
    CHECK(s(2) == 40.0);
}

// ============================================================
// Matrix row / col slicing
// ============================================================

TEST_CASE("dynamic_matrix_row_col", "[dynamic][slicing]") {
    MatrixXX<double> A(3, 4);
    for (index_type j = 0; j < 3; ++j) {
        for (index_type k = 0; k < 4; ++k) { A(j, k) = 4.0 * j + k; }
    }

    auto r1 = A.row(index_type(1));
    REQUIRE(r1.size() == 4);
    CHECK(r1(0) == 4.0);
    CHECK(r1(1) == 5.0);
    CHECK(r1(2) == 6.0);
    CHECK(r1(3) == 7.0);

    auto c2 = A.col(index_type(2));
    REQUIRE(c2.size() == 3);
    CHECK(c2(0) == 2.0);
    CHECK(c2(1) == 6.0);
    CHECK(c2(2) == 10.0);
}

// ============================================================
// Matrix blocks: topRows, bottomRows, leftCols, rightCols
// ============================================================

TEST_CASE("dynamic_matrix_topRows", "[dynamic][slicing][blocks]") {
    MatrixXX<double> A(4, 3);
    for (index_type j = 0; j < 4; ++j) {
        for (index_type k = 0; k < 3; ++k) { A(j, k) = j; }
    }

    auto top = A.topRows(2);
    REQUIRE(top.rows() == 2);
    REQUIRE(top.cols() == 3);
    for (index_type j = 0; j < 2; ++j) {
        for (index_type k = 0; k < 3; ++k) { CHECK(top(j, k) == j); }
    }
}

TEST_CASE("dynamic_matrix_bottomRows", "[dynamic][slicing][blocks]") {
    MatrixXX<double> A(4, 3);
    for (index_type j = 0; j < 4; ++j) {
        for (index_type k = 0; k < 3; ++k) { A(j, k) = j; }
    }

    auto bot = A.bottomRows(2);
    REQUIRE(bot.rows() == 2);
    REQUIRE(bot.cols() == 3);
    CHECK(bot(0, 0) == 2.0);
    CHECK(bot(1, 0) == 3.0);
}

TEST_CASE("dynamic_matrix_leftCols", "[dynamic][slicing][blocks]") {
    MatrixXX<double> A(3, 5);
    for (index_type j = 0; j < 3; ++j) {
        for (index_type k = 0; k < 5; ++k) { A(j, k) = k; }
    }

    auto left = A.leftCols(3);
    REQUIRE(left.rows() == 3);
    REQUIRE(left.cols() == 3);
    for (index_type j = 0; j < 3; ++j) {
        for (index_type k = 0; k < 3; ++k) { CHECK(left(j, k) == k); }
    }
}

TEST_CASE("dynamic_matrix_rightCols", "[dynamic][slicing][blocks]") {
    MatrixXX<double> A(3, 5);
    for (index_type j = 0; j < 3; ++j) {
        for (index_type k = 0; k < 5; ++k) { A(j, k) = k; }
    }

    auto right = A.rightCols(3);
    REQUIRE(right.rows() == 3);
    REQUIRE(right.cols() == 3);
    for (index_type j = 0; j < 3; ++j) {
        for (index_type k = 0; k < 3; ++k) { CHECK(right(j, k) == k + 2); }
    }
}

// ============================================================
// Diagonal access
// ============================================================

TEST_CASE("dynamic_diagonal_access", "[dynamic][diagonal]") {
    MatrixXX<double> A(3, 3);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;
    A(2, 0) = 7;
    A(2, 1) = 8;
    A(2, 2) = 9;

    auto d = A.diagonal();
    REQUIRE(d.size() == 3);
    CHECK(d(0) == 1.0);
    CHECK(d(1) == 5.0);
    CHECK(d(2) == 9.0);
}

TEST_CASE("dynamic_diagonal_assignment", "[dynamic][diagonal]") {
    MatrixXX<double> A(3, 3);
    for (index_type j = 0; j < 3; ++j) {
        for (index_type k = 0; k < 3; ++k) { A(j, k) = 0; }
    }

    A.diagonal() = expression::nullary::Constant<double, 3>(1.0);
    CHECK(A(0, 0) == 1.0);
    CHECK(A(1, 1) == 1.0);
    CHECK(A(2, 2) == 1.0);
    CHECK(A(0, 1) == 0.0);
    CHECK(A.trace() == 3.0);
}

// ============================================================
// Colwise / rowwise partial norm
// ============================================================

TEST_CASE("dynamic_colwise_rowwise_norm", "[dynamic][reduction][partial]") {
    MatrixXX<double> A(2, 3);
    A(0, 0) = 3;
    A(0, 1) = 0;
    A(0, 2) = 4;
    A(1, 0) = 0;
    A(1, 1) = 5;
    A(1, 2) = 0;

    auto cn = A.colwise().norm();
    REQUIRE(cn.size() == 3);
    CHECK(cn(0) == Catch::Approx(3.0));
    CHECK(cn(1) == Catch::Approx(5.0));
    CHECK(cn(2) == Catch::Approx(4.0));

    auto rn = A.rowwise().norm();
    REQUIRE(rn.size() == 2);
    CHECK(rn(0) == Catch::Approx(5.0));
    CHECK(rn(1) == Catch::Approx(5.0));
}

// ============================================================
// Unit vector construction
// ============================================================

TEST_CASE("dynamic_unit_vector", "[dynamic][nullary]") {
    VectorBase e0 = expression::nullary::unit_vector<double>(3, 0);
    VectorBase e1 = expression::nullary::unit_vector<double>(3, 1);
    VectorBase e2 = expression::nullary::unit_vector<double>(3, 2);

    REQUIRE(e0.size() == 3);
    CHECK(e0(0) == 1.0);
    CHECK(e0(1) == 0.0);
    CHECK(e0(2) == 0.0);

    CHECK(e1(0) == 0.0);
    CHECK(e1(1) == 1.0);
    CHECK(e1(2) == 0.0);

    CHECK(e2(0) == 0.0);
    CHECK(e2(1) == 0.0);
    CHECK(e2(2) == 1.0);
}

TEST_CASE("dynamic_dot_unit_vector", "[dynamic][reduction]") {
    VectorX<double> v = {2.0, 5.0, 9.0};
    VectorBase e0 = expression::nullary::unit_vector<double>(3, 0);
    VectorBase e1 = expression::nullary::unit_vector<double>(3, 1);
    VectorBase e2 = expression::nullary::unit_vector<double>(3, 2);

    CHECK(v.dot(e0) == 2.0);
    CHECK(v.dot(e1) == 5.0);
    CHECK(v.dot(e2) == 9.0);
}

// ============================================================
// Homogeneous coordinates
// ============================================================

TEST_CASE("dynamic_homogeneous_position", "[dynamic][homogeneous]") {
    VectorX<double> v = {1.0, 2.0, 3.0};

    auto h = expression::unary::homogeneous_position(v.expression());
    REQUIRE(h.extent(0) == 4);
    CHECK(h(0) == 1.0);
    CHECK(h(1) == 2.0);
    CHECK(h(2) == 3.0);
    CHECK(h(3) == 1.0);
}

TEST_CASE("dynamic_homogeneous_vector", "[dynamic][homogeneous]") {
    VectorX<double> v = {1.0, 2.0, 3.0};

    auto h = expression::unary::homogeneous_vector(v.expression());
    REQUIRE(h.extent(0) == 4);
    CHECK(h(0) == 1.0);
    CHECK(h(1) == 2.0);
    CHECK(h(2) == 3.0);
    CHECK(h(3) == 0.0);
}

// ============================================================
// Custom unary_expr (lambda)
// ============================================================

TEST_CASE("dynamic_unary_expr", "[dynamic][unary]") {
    VectorX<int> v = {3, 5, 6, 9, 10};

    auto result = v.unary_expr([](int x) -> bool { return x % 3 == 0; });
    REQUIRE(result.size() == 5);
    CHECK(result(0) == true);
    CHECK(result(1) == false);
    CHECK(result(2) == true);
    CHECK(result(3) == true);
    CHECK(result(4) == false);
}

// ============================================================
// Reshape 2D-to-1D and 2D-to-2D
// ============================================================

TEST_CASE("dynamic_reshape_matrix_to_vec", "[dynamic][reshape]") {
    MatrixXX<double> A(2, 3);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;

    using new_extents = zipper::extents<6>;
    auto r = expression::unary::Reshape(A.expression(), new_extents{});
    REQUIRE(r.extent(0) == 6);
    CHECK(r(0) == 1.0);
    CHECK(r(1) == 2.0);
    CHECK(r(2) == 3.0);
    CHECK(r(3) == 4.0);
    CHECK(r(4) == 5.0);
    CHECK(r(5) == 6.0);
}

TEST_CASE("dynamic_reshape_matrix_to_matrix", "[dynamic][reshape]") {
    MatrixXX<double> A(2, 3);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;

    using new_extents = zipper::extents<3, 2>;
    auto r = expression::unary::Reshape(A.expression(), new_extents{});
    REQUIRE(r.extent(0) == 3);
    REQUIRE(r.extent(1) == 2);
    CHECK(r(0, 0) == 1.0);
    CHECK(r(0, 1) == 2.0);
    CHECK(r(1, 0) == 3.0);
    CHECK(r(1, 1) == 4.0);
    CHECK(r(2, 0) == 5.0);
    CHECK(r(2, 1) == 6.0);
}

// ============================================================
// Scalar division
// ============================================================

TEST_CASE("dynamic_scalar_divide_vector", "[dynamic][scalar]") {
    VectorX<double> x = {6.0, 9.0, 15.0};

    auto y = x / 3.0;
    REQUIRE(y.size() == 3);
    CHECK(y(0) == 2.0);
    CHECK(y(1) == 3.0);
    CHECK(y(2) == 5.0);
}

TEST_CASE("dynamic_scalar_subtract_array", "[dynamic][scalar]") {
    VectorX<double> x = {5.0, 10.0, 15.0};

    auto y = x.as_array() - 2.0;
    CHECK(y(0) == 3.0);
    CHECK(y(1) == 8.0);
    CHECK(y(2) == 13.0);
}

// ============================================================
// Compound Assignment
// ============================================================

TEST_CASE("dynamic_vector_compound_add", "[dynamic][compound]") {
    VectorX<double> x = {1.0, 2.0, 3.0};
    VectorX<double> y = {10.0, 20.0, 30.0};
    x += y;
    CHECK(x(0) == 11.0);
    CHECK(x(1) == 22.0);
    CHECK(x(2) == 33.0);
}

TEST_CASE("dynamic_vector_compound_sub", "[dynamic][compound]") {
    VectorX<double> x = {10.0, 20.0, 30.0};
    VectorX<double> y = {1.0, 2.0, 3.0};
    x -= y;
    CHECK(x(0) == 9.0);
    CHECK(x(1) == 18.0);
    CHECK(x(2) == 27.0);
}

TEST_CASE("dynamic_vector_compound_scalar_mul", "[dynamic][compound]") {
    VectorX<double> x = {1.0, 2.0, 3.0};
    x *= 3.0;
    CHECK(x(0) == 3.0);
    CHECK(x(1) == 6.0);
    CHECK(x(2) == 9.0);
}

TEST_CASE("dynamic_vector_compound_scalar_div", "[dynamic][compound]") {
    VectorX<double> x = {6.0, 12.0, 18.0};
    x /= 3.0;
    CHECK(x(0) == 2.0);
    CHECK(x(1) == 4.0);
    CHECK(x(2) == 6.0);
}

TEST_CASE("dynamic_matrix_compound_add", "[dynamic][compound]") {
    MatrixXX<double> A(2, 2);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 0) = 3.0;
    A(1, 1) = 4.0;
    MatrixXX<double> B(2, 2);
    B(0, 0) = 10.0;
    B(0, 1) = 20.0;
    B(1, 0) = 30.0;
    B(1, 1) = 40.0;
    A += B;
    CHECK(A(0, 0) == 11.0);
    CHECK(A(0, 1) == 22.0);
    CHECK(A(1, 0) == 33.0);
    CHECK(A(1, 1) == 44.0);
}

TEST_CASE("dynamic_matrix_compound_sub", "[dynamic][compound]") {
    MatrixXX<double> A(2, 2);
    A(0, 0) = 10.0;
    A(0, 1) = 20.0;
    A(1, 0) = 30.0;
    A(1, 1) = 40.0;
    MatrixXX<double> B(2, 2);
    B(0, 0) = 1.0;
    B(0, 1) = 2.0;
    B(1, 0) = 3.0;
    B(1, 1) = 4.0;
    A -= B;
    CHECK(A(0, 0) == 9.0);
    CHECK(A(0, 1) == 18.0);
    CHECK(A(1, 0) == 27.0);
    CHECK(A(1, 1) == 36.0);
}

TEST_CASE("dynamic_matrix_compound_scalar_mul", "[dynamic][compound]") {
    MatrixXX<double> A(2, 2);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 0) = 3.0;
    A(1, 1) = 4.0;
    A *= 2.0;
    CHECK(A(0, 0) == 2.0);
    CHECK(A(0, 1) == 4.0);
    CHECK(A(1, 0) == 6.0);
    CHECK(A(1, 1) == 8.0);
}

TEST_CASE("dynamic_matrix_compound_scalar_div", "[dynamic][compound]") {
    MatrixXX<double> A(2, 2);
    A(0, 0) = 2.0;
    A(0, 1) = 4.0;
    A(1, 0) = 6.0;
    A(1, 1) = 8.0;
    A /= 2.0;
    CHECK(A(0, 0) == 1.0);
    CHECK(A(0, 1) == 2.0);
    CHECK(A(1, 0) == 3.0);
    CHECK(A(1, 1) == 4.0);
}

// NOTE: Form compound assignment (*=, /=) is currently broken because
// FormBase::operator= does `expression() = v.expression()` which calls
// MDArray::operator= (only accepts MDArray, not expression types).
// This should be fixed by using Base::operator= or expression().assign()
// like VectorBase/MatrixBase do. Tracked separately.

// ============================================================
// Normalized
// ============================================================

TEST_CASE("dynamic_vector_normalized", "[dynamic][reduction]") {
    VectorX<double> x = {3.0, 4.0};

    auto n = x.normalized();
    REQUIRE(n.size() == 2);
    CHECK(n(0) == Catch::Approx(0.6));
    CHECK(n(1) == Catch::Approx(0.8));
}
