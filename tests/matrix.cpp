

#include <iostream>
#include <zipper/ArrayBase.hxx>
#include <zipper/Matrix.hpp>
#include <zipper/MatrixBase.hxx>
#include <zipper/Vector.hpp>
#include <zipper/VectorBase.hxx>
#include <zipper/expression/nullary/Random.hpp>

#include "catch_include.hpp"

namespace {
void print(zipper::concepts::Matrix auto const &M) {
    for (zipper::index_type j = 0; j < M.extent(0); ++j) {
        for (zipper::index_type k = 0; k < M.extent(1); ++k) {
            std::cout << M(j, k) << " ";
        }
        std::cout << std::endl;
    }
}
}// namespace

TEST_CASE("test_assignment", "[matrix][storage][dense]") {
    zipper::Matrix<double, 3, std::dynamic_extent> M(3);
    zipper::Matrix<double, 3, 5> N;
    zipper::Matrix<double, std::dynamic_extent, std::dynamic_extent> O(2, 2);
    zipper::Vector<double, 3> x;

    O(0, 0) = 0;
    O(1, 0) = 1;
    O(0, 1) = 3;
    O(1, 1) = 4;

    CHECK(O(0, 0) == 0);
    CHECK(O(1, 0) == 1);
    CHECK(O(0, 1) == 3);
    CHECK(O(1, 1) == 4);

    x(0) = 2;
    x(1) = 5;
    x(2) = 9;
    M(0, 0) = 0;
    M(1, 0) = 1;
    M(2, 0) = 2;
    M(0, 1) = 3;
    M(1, 1) = 4;
    M(2, 1) = 5;
    M(0, 2) = 6;
    M(1, 2) = 7;
    M(2, 2) = 8;

    N(0, 0) = 0;
    N(1, 0) = 1;
    N(2, 0) = 2;
    N(0, 1) = 3;
    N(1, 1) = 4;
    N(2, 1) = 5;
    N(0, 2) = 6;
    N(1, 2) = 7;
    N(2, 2) = 8;
    N(0, 3) = 9;
    N(1, 3) = 10;
    N(2, 3) = 11;
    N(0, 4) = 12;
    N(1, 4) = 13;
    N(2, 4) = 14;

    CHECK(x(0) == 2);
    CHECK(x(1) == 5);
    CHECK(x(2) == 9);
    CHECK(M(0, 0) == 0);
    CHECK(M(1, 0) == 1);
    CHECK(M(2, 0) == 2);
    CHECK(M(0, 1) == 3);
    CHECK(M(1, 1) == 4);
    CHECK(M(2, 1) == 5);
    CHECK(M(0, 2) == 6);
    CHECK(M(1, 2) == 7);
    CHECK(M(2, 2) == 8);
    CHECK(N(0, 0) == 0);
    CHECK(N(1, 0) == 1);
    CHECK(N(2, 0) == 2);
    CHECK(N(0, 1) == 3);
    CHECK(N(1, 1) == 4);
    CHECK(N(2, 1) == 5);
    CHECK(N(0, 2) == 6);
    CHECK(N(1, 2) == 7);
    CHECK(N(2, 2) == 8);
    CHECK(N(0, 3) == 9);
    CHECK(N(1, 3) == 10);
    CHECK(N(2, 3) == 11);
    CHECK(N(0, 4) == 12);
    CHECK(N(1, 4) == 13);
    CHECK(N(2, 4) == 14);
}

TEST_CASE("test_matrix_eval", "[matrix][storage][dense]") {
    zipper::Matrix<double, 3, 5> N =
      zipper::expression::nullary::uniform_random<double>({});

    auto x = (N * 2).eval();
    auto v = N.as_array();
    STATIC_CHECK(std::is_same_v<std::decay_t<decltype(v)>::extents_type,
                                 decltype(N)::extents_type>);
    STATIC_CHECK(std::is_same_v<decltype(v.extents()), decltype(N.extents())>);
    STATIC_CHECK(std::decay_t<decltype(v)>::extents_type::rank() == 2);
    // auto y = N.as_array().eval();
    print(x);
}

TEST_CASE("test_span", "[matrix][storage][dense]") {
    zipper::Matrix<zipper::index_type, 3, 6> C;
    zipper::Matrix<zipper::index_type, 3, std::dynamic_extent> RC(6);
    zipper::Matrix<zipper::index_type, std::dynamic_extent, std::dynamic_extent>
      R(3, 6);

    auto CS = C.as_span();
    auto RCS = RC.as_span();
    auto RS = R.as_span();

    REQUIRE(CS.extents() == C.extents());
    REQUIRE(RS.extents() == R.extents());
    REQUIRE(RCS.extents() == RC.extents());

    for (zipper::index_type j = 0; j < 3; ++j) {
        for (zipper::index_type k = 0; k < 6; ++k) {
            CHECK(&C(j, k) == &CS(j, k));
            CHECK(&RC(j, k) == &RCS(j, k));
            CHECK(&R(j, k) == &RS(j, k));
        }
    }
}

TEST_CASE("test_matrix_span", "[matrix][storage][dense][span]") {
    std::vector<int> vec = { 0, 1, 2, 3 };
    zipper::Matrix<int, 2, 2>::span_type M = std::span<int, 4>(vec);
    zipper::Matrix<int, std::dynamic_extent, std::dynamic_extent>::span_type Md(
      std::span<int>(vec), zipper::create_dextents(2, 2));

    STATIC_CHECK(M.static_extent(0) == 2);
    STATIC_CHECK(M.static_extent(1) == 2);
    STATIC_CHECK(Md.static_extent(0) == std::dynamic_extent);
    STATIC_CHECK(Md.static_extent(1) == std::dynamic_extent);
    REQUIRE(M.extent(0) == 2);
    REQUIRE(M.extent(1) == 2);
    REQUIRE(Md.extent(0) == 2);
    REQUIRE(Md.extent(1) == 2);

    CHECK(M(0, 0) == 0);
    CHECK(M(0, 1) == 1);
    CHECK(M(1, 0) == 2);
    CHECK(M(1, 1) == 3);

    CHECK((M == Md));

    {
        zipper::Matrix m = M;
        auto a = m.as_span();
        auto b = m.as_const_span();
        CHECK((a == b));

        zipper::Matrix<int, 2, 2>::const_span_type d = a;
        CHECK((a == d));
    }

    // this last case WOULD be very cool, but seems to not work due to a parse
    // limitation in type deductions? In particular, gcc at least seems to
    // really want y to be the name of a variable of type VectorBase
    // VectorBase(y) = {4, 5};
    // CHECK(v(0) == 2);
    // CHECK(v(1) == 3);
}
// ============================================================
// Matrix negate
// ============================================================

TEST_CASE("matrix_negate", "[matrix][unary]") {
    zipper::Matrix<double, 2, 2> M;
    M(0, 0) = 1.0;
    M(0, 1) = 2.0;
    M(1, 0) = 3.0;
    M(1, 1) = 4.0;

    auto N = -M;
    CHECK(N(0, 0) == -1.0);
    CHECK(N(0, 1) == -2.0);
    CHECK(N(1, 0) == -3.0);
    CHECK(N(1, 1) == -4.0);
}

// ============================================================
// Matrix scalar ops
// ============================================================

TEST_CASE("matrix_scalar_multiply", "[matrix][scalar]") {
    zipper::Matrix<double, 2, 2> M;
    M(0, 0) = 1.0;
    M(0, 1) = 2.0;
    M(1, 0) = 3.0;
    M(1, 1) = 4.0;

    auto A = M * 3.0;
    CHECK(A(0, 0) == 3.0);
    CHECK(A(0, 1) == 6.0);
    CHECK(A(1, 0) == 9.0);
    CHECK(A(1, 1) == 12.0);

    auto B = 3.0 * M;
    CHECK(B(0, 0) == 3.0);
    CHECK(B(0, 1) == 6.0);
    CHECK(B(1, 0) == 9.0);
    CHECK(B(1, 1) == 12.0);
}

TEST_CASE("matrix_scalar_divide", "[matrix][scalar]") {
    zipper::Matrix<double, 2, 2> M;
    M(0, 0) = 10.0;
    M(0, 1) = 20.0;
    M(1, 0) = 30.0;
    M(1, 1) = 40.0;

    auto A = M / 2.0;
    CHECK(A(0, 0) == 5.0);
    CHECK(A(0, 1) == 10.0);
    CHECK(A(1, 0) == 15.0);
    CHECK(A(1, 1) == 20.0);
}

// ============================================================
// Matrix addition / subtraction
// ============================================================

TEST_CASE("matrix_add", "[matrix][binary]") {
    zipper::Matrix<double, 2, 2> A;
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 0) = 3.0;
    A(1, 1) = 4.0;

    zipper::Matrix<double, 2, 2> B;
    B(0, 0) = 10.0;
    B(0, 1) = 20.0;
    B(1, 0) = 30.0;
    B(1, 1) = 40.0;

    auto C = A + B;
    CHECK(C(0, 0) == 11.0);
    CHECK(C(0, 1) == 22.0);
    CHECK(C(1, 0) == 33.0);
    CHECK(C(1, 1) == 44.0);
}

TEST_CASE("matrix_subtract", "[matrix][binary]") {
    zipper::Matrix<double, 2, 2> A;
    A(0, 0) = 10.0;
    A(0, 1) = 20.0;
    A(1, 0) = 30.0;
    A(1, 1) = 40.0;

    zipper::Matrix<double, 2, 2> B;
    B(0, 0) = 1.0;
    B(0, 1) = 2.0;
    B(1, 0) = 3.0;
    B(1, 1) = 4.0;

    auto C = A - B;
    CHECK(C(0, 0) == 9.0);
    CHECK(C(0, 1) == 18.0);
    CHECK(C(1, 0) == 27.0);
    CHECK(C(1, 1) == 36.0);
}

// ============================================================
// Multi-op chains: 2*A + 3*B
// ============================================================

TEST_CASE("matrix_multi_op_chain", "[matrix][chain]") {
    zipper::Matrix<double, 2, 2> A;
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 0) = 3.0;
    A(1, 1) = 4.0;

    zipper::Matrix<double, 2, 2> B;
    B(0, 0) = 10.0;
    B(0, 1) = 20.0;
    B(1, 0) = 30.0;
    B(1, 1) = 40.0;

    auto C = 2.0 * A + 3.0 * B;
    // 2*[1,2;3,4] + 3*[10,20;30,40] = [2,4;6,8] + [30,60;90,120] = [32,64;96,128]
    CHECK(C(0, 0) == 32.0);
    CHECK(C(0, 1) == 64.0);
    CHECK(C(1, 0) == 96.0);
    CHECK(C(1, 1) == 128.0);
}

// ============================================================
// Matrix-matrix product
// ============================================================

TEST_CASE("matrix_product_identity", "[matrix][product]") {
    zipper::Matrix<double, 2, 2> I;
    I(0, 0) = 1.0;
    I(0, 1) = 0.0;
    I(1, 0) = 0.0;
    I(1, 1) = 1.0;

    zipper::Matrix<double, 2, 2> A;
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 0) = 3.0;
    A(1, 1) = 4.0;

    auto C = I * A;
    CHECK(C(0, 0) == 1.0);
    CHECK(C(0, 1) == 2.0);
    CHECK(C(1, 0) == 3.0);
    CHECK(C(1, 1) == 4.0);
}

TEST_CASE("matrix_product_general", "[matrix][product]") {
    // [1 2] * [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
    // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
    zipper::Matrix<double, 2, 2> A;
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 0) = 3.0;
    A(1, 1) = 4.0;

    zipper::Matrix<double, 2, 2> B;
    B(0, 0) = 5.0;
    B(0, 1) = 6.0;
    B(1, 0) = 7.0;
    B(1, 1) = 8.0;

    auto C = A * B;
    CHECK(C(0, 0) == 19.0);
    CHECK(C(0, 1) == 22.0);
    CHECK(C(1, 0) == 43.0);
    CHECK(C(1, 1) == 50.0);
}

TEST_CASE("matrix_product_nonsquare", "[matrix][product]") {
    // [1 2 3] * [1 0]   = [1+4+9   0+2+6]  = [14  8]
    // [4 5 6]   [2 1]     [4+10+18 0+5+12]    [32 17]
    //           [3 2]
    zipper::Matrix<double, 2, 3> A;
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(0, 2) = 3.0;
    A(1, 0) = 4.0;
    A(1, 1) = 5.0;
    A(1, 2) = 6.0;

    zipper::Matrix<double, 3, 2> B;
    B(0, 0) = 1.0;
    B(0, 1) = 0.0;
    B(1, 0) = 2.0;
    B(1, 1) = 1.0;
    B(2, 0) = 3.0;
    B(2, 1) = 2.0;

    auto C = A * B;
    STATIC_CHECK(std::decay_t<decltype(C)>::extents_type::static_extent(0) == 2);
    STATIC_CHECK(std::decay_t<decltype(C)>::extents_type::static_extent(1) == 2);
    CHECK(C(0, 0) == 14.0);
    CHECK(C(0, 1) == 8.0);
    CHECK(C(1, 0) == 32.0);
    CHECK(C(1, 1) == 17.0);
}

// ============================================================
// Matrix-vector product
// ============================================================

TEST_CASE("matrix_vector_product", "[matrix][product]") {
    // [1 2] * [5] = [1*5+2*6] = [17]
    // [3 4]   [6]   [3*5+4*6]   [39]
    zipper::Matrix<double, 2, 2> M;
    M(0, 0) = 1.0;
    M(0, 1) = 2.0;
    M(1, 0) = 3.0;
    M(1, 1) = 4.0;

    zipper::Vector<double, 2> x{ 5.0, 6.0 };

    auto y = M * x;
    STATIC_CHECK(std::decay_t<decltype(y)>::extents_type::rank() == 1);
    CHECK(y(0) == 17.0);
    CHECK(y(1) == 39.0);
}

TEST_CASE("matrix_vector_product_nonsquare", "[matrix][product]") {
    // [1 2 3] * [1] = [1+4+9]  = [14]
    // [4 5 6]   [2]   [4+10+18]   [32]
    //           [3]
    zipper::Matrix<double, 2, 3> M;
    M(0, 0) = 1.0;
    M(0, 1) = 2.0;
    M(0, 2) = 3.0;
    M(1, 0) = 4.0;
    M(1, 1) = 5.0;
    M(1, 2) = 6.0;

    zipper::Vector<double, 3> x{ 1.0, 2.0, 3.0 };

    auto y = M * x;
    STATIC_CHECK(std::decay_t<decltype(y)>::extents_type::static_extent(0) == 2);
    CHECK(y(0) == 14.0);
    CHECK(y(1) == 32.0);
}

TEST_CASE("matrix_vector_product_identity", "[matrix][product]") {
    zipper::Matrix<double, 3, 3> I;
    I(0, 0) = 1.0;
    I(0, 1) = 0.0;
    I(0, 2) = 0.0;
    I(1, 0) = 0.0;
    I(1, 1) = 1.0;
    I(1, 2) = 0.0;
    I(2, 0) = 0.0;
    I(2, 1) = 0.0;
    I(2, 2) = 1.0;

    zipper::Vector<double, 3> x{ 7.0, 8.0, 9.0 };

    auto y = I * x;
    CHECK(y(0) == 7.0);
    CHECK(y(1) == 8.0);
    CHECK(y(2) == 9.0);
}

// ============================================================
// Matrix transpose
// ============================================================

TEST_CASE("matrix_transpose", "[matrix][unary]") {
    zipper::Matrix<double, 2, 3> M;
    M(0, 0) = 1.0;
    M(0, 1) = 2.0;
    M(0, 2) = 3.0;
    M(1, 0) = 4.0;
    M(1, 1) = 5.0;
    M(1, 2) = 6.0;

    auto T = M.transpose();
    STATIC_CHECK(std::decay_t<decltype(T)>::extents_type::static_extent(0) == 3);
    STATIC_CHECK(std::decay_t<decltype(T)>::extents_type::static_extent(1) == 2);
    CHECK(T(0, 0) == 1.0);
    CHECK(T(1, 0) == 2.0);
    CHECK(T(2, 0) == 3.0);
    CHECK(T(0, 1) == 4.0);
    CHECK(T(1, 1) == 5.0);
    CHECK(T(2, 1) == 6.0);
}

// ============================================================
// Matrix trace
// ============================================================

TEST_CASE("matrix_trace", "[matrix][reduction]") {
    zipper::Matrix<double, 3, 3> M;
    M(0, 0) = 1.0;
    M(0, 1) = 0.0;
    M(0, 2) = 0.0;
    M(1, 0) = 0.0;
    M(1, 1) = 2.0;
    M(1, 2) = 0.0;
    M(2, 0) = 0.0;
    M(2, 1) = 0.0;
    M(2, 2) = 3.0;
    CHECK(M.trace() == 6.0);
}

// ============================================================
// Matrix diagonal
// ============================================================

TEST_CASE("matrix_diagonal_read", "[matrix][view]") {
    zipper::Matrix<double, 3, 3> M;
    M(0, 0) = 10.0;
    M(0, 1) = 0.0;
    M(0, 2) = 0.0;
    M(1, 0) = 0.0;
    M(1, 1) = 20.0;
    M(1, 2) = 0.0;
    M(2, 0) = 0.0;
    M(2, 1) = 0.0;
    M(2, 2) = 30.0;

    auto d = M.diagonal();
    CHECK(d(0) == 10.0);
    CHECK(d(1) == 20.0);
    CHECK(d(2) == 30.0);
}

// ============================================================
// Matrix transpose write-through
// ============================================================

TEST_CASE("matrix_transpose_write_through", "[matrix][view][mutable]") {
    zipper::Matrix<double, 2, 2> M{ { 1.0, 2.0 }, { 3.0, 4.0 } };

    M.transpose()(1, 0) = 55.0;
    CHECK(M(0, 1) == 55.0);
    CHECK(M(1, 0) == 3.0);// unchanged
}

// ============================================================
// Matrix row/col slicing
// ============================================================

TEST_CASE("matrix_row_col_access", "[matrix][view]") {
    zipper::Matrix<double, 2, 3> M;
    M(0, 0) = 1.0;
    M(0, 1) = 2.0;
    M(0, 2) = 3.0;
    M(1, 0) = 4.0;
    M(1, 1) = 5.0;
    M(1, 2) = 6.0;

    // Row 0
    auto r0 = M.row(0);
    CHECK(r0(0) == 1.0);
    CHECK(r0(1) == 2.0);
    CHECK(r0(2) == 3.0);

    // Col 1
    auto c1 = M.col(1);
    CHECK(c1(0) == 2.0);
    CHECK(c1(1) == 5.0);
}

// ============================================================
// Matrix equality
// ============================================================

TEST_CASE("matrix_equality", "[matrix][comparison]") {
    zipper::Matrix<double, 2, 2> A;
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 0) = 3.0;
    A(1, 1) = 4.0;

    zipper::Matrix<double, 2, 2> B;
    B(0, 0) = 1.0;
    B(0, 1) = 2.0;
    B(1, 0) = 3.0;
    B(1, 1) = 4.0;

    zipper::Matrix<double, 2, 2> C;
    C(0, 0) = 1.0;
    C(0, 1) = 2.0;
    C(1, 0) = 3.0;
    C(1, 1) = 999.0;

    CHECK(A == B);
    CHECK(A != C);
}

// ============================================================
// Block operations: topRows, bottomRows, leftCols, rightCols
// Tests both dynamic and template<N> forms
// ============================================================

TEST_CASE("block_topRows_dynamic", "[matrix][block]") {
    // 6×4 matrix, values = row index
    zipper::Matrix<int, 6, 4> M;
    for (zipper::index_type j = 0; j < 6; ++j)
        for (zipper::index_type k = 0; k < 4; ++k)
            M(j, k) = static_cast<int>(j);

    auto T = M.topRows(3);
    REQUIRE(T.extent(0) == 3);
    REQUIRE(T.extent(1) == 4);
    for (zipper::index_type j = 0; j < 3; ++j)
        for (zipper::index_type k = 0; k < 4; ++k)
            CHECK(T(j, k) == static_cast<int>(j));
}

TEST_CASE("block_topRows_template", "[matrix][block]") {
    zipper::Matrix<int, 6, 4> M;
    for (zipper::index_type j = 0; j < 6; ++j)
        for (zipper::index_type k = 0; k < 4; ++k)
            M(j, k) = static_cast<int>(j);

    auto T = M.template topRows<3>();
    STATIC_CHECK(std::decay_t<decltype(T)>::extents_type::static_extent(0) == 3);
    REQUIRE(T.extent(0) == 3);
    REQUIRE(T.extent(1) == 4);
    for (zipper::index_type j = 0; j < 3; ++j)
        for (zipper::index_type k = 0; k < 4; ++k)
            CHECK(T(j, k) == static_cast<int>(j));
}

TEST_CASE("block_bottomRows_dynamic", "[matrix][block]") {
    // 6×4 matrix, values = row index
    zipper::Matrix<int, 6, 4> M;
    for (zipper::index_type j = 0; j < 6; ++j)
        for (zipper::index_type k = 0; k < 4; ++k)
            M(j, k) = static_cast<int>(j);

    auto B = M.bottomRows(3);
    REQUIRE(B.extent(0) == 3);
    REQUIRE(B.extent(1) == 4);
    // Bottom 3 rows of a 6-row matrix: rows 3,4,5
    for (zipper::index_type j = 0; j < 3; ++j)
        for (zipper::index_type k = 0; k < 4; ++k)
            CHECK(B(j, k) == static_cast<int>(j + 3));
}

TEST_CASE("block_bottomRows_template", "[matrix][block]") {
    // This is the codepath that had the bug (operands swapped in offset calc)
    zipper::Matrix<int, 6, 4> M;
    for (zipper::index_type j = 0; j < 6; ++j)
        for (zipper::index_type k = 0; k < 4; ++k)
            M(j, k) = static_cast<int>(j);

    auto B = M.template bottomRows<3>();
    STATIC_CHECK(std::decay_t<decltype(B)>::extents_type::static_extent(0) == 3);
    REQUIRE(B.extent(0) == 3);
    REQUIRE(B.extent(1) == 4);
    // Bottom 3 rows of a 6-row matrix: rows 3,4,5
    for (zipper::index_type j = 0; j < 3; ++j)
        for (zipper::index_type k = 0; k < 4; ++k)
            CHECK(B(j, k) == static_cast<int>(j + 3));
}

TEST_CASE("block_bottomRows_template_all_rows", "[matrix][block]") {
    // Edge case: bottomRows<N> where N == total rows
    zipper::Matrix<int, 3, 2> M;
    M(0, 0) = 10;
    M(0, 1) = 11;
    M(1, 0) = 20;
    M(1, 1) = 21;
    M(2, 0) = 30;
    M(2, 1) = 31;

    auto B = M.template bottomRows<3>();
    REQUIRE(B.extent(0) == 3);
    CHECK(B(0, 0) == 10);
    CHECK(B(1, 0) == 20);
    CHECK(B(2, 0) == 30);
}

TEST_CASE("block_bottomRows_template_one_row", "[matrix][block]") {
    // Edge case: bottomRows<1>
    zipper::Matrix<int, 4, 3> M;
    for (zipper::index_type j = 0; j < 4; ++j)
        for (zipper::index_type k = 0; k < 3; ++k)
            M(j, k) = static_cast<int>(j * 10 + k);

    auto B = M.template bottomRows<1>();
    REQUIRE(B.extent(0) == 1);
    REQUIRE(B.extent(1) == 3);
    CHECK(B(0, 0) == 30);
    CHECK(B(0, 1) == 31);
    CHECK(B(0, 2) == 32);
}

TEST_CASE("block_leftCols_dynamic", "[matrix][block]") {
    // 3×6 matrix, values = col index
    zipper::Matrix<int, 3, 6> M;
    for (zipper::index_type j = 0; j < 3; ++j)
        for (zipper::index_type k = 0; k < 6; ++k)
            M(j, k) = static_cast<int>(k);

    auto L = M.leftCols(4);
    REQUIRE(L.extent(0) == 3);
    REQUIRE(L.extent(1) == 4);
    for (zipper::index_type j = 0; j < 3; ++j)
        for (zipper::index_type k = 0; k < 4; ++k)
            CHECK(L(j, k) == static_cast<int>(k));
}

TEST_CASE("block_leftCols_template", "[matrix][block]") {
    zipper::Matrix<int, 3, 6> M;
    for (zipper::index_type j = 0; j < 3; ++j)
        for (zipper::index_type k = 0; k < 6; ++k)
            M(j, k) = static_cast<int>(k);

    auto L = M.template leftCols<4>();
    STATIC_CHECK(std::decay_t<decltype(L)>::extents_type::static_extent(1) == 4);
    REQUIRE(L.extent(0) == 3);
    REQUIRE(L.extent(1) == 4);
    for (zipper::index_type j = 0; j < 3; ++j)
        for (zipper::index_type k = 0; k < 4; ++k)
            CHECK(L(j, k) == static_cast<int>(k));
}

TEST_CASE("block_rightCols_dynamic", "[matrix][block]") {
    // 3×6 matrix, values = col index
    zipper::Matrix<int, 3, 6> M;
    for (zipper::index_type j = 0; j < 3; ++j)
        for (zipper::index_type k = 0; k < 6; ++k)
            M(j, k) = static_cast<int>(k);

    auto R = M.rightCols(4);
    REQUIRE(R.extent(0) == 3);
    REQUIRE(R.extent(1) == 4);
    // Right 4 cols of a 6-col matrix: cols 2,3,4,5
    for (zipper::index_type j = 0; j < 3; ++j)
        for (zipper::index_type k = 0; k < 4; ++k)
            CHECK(R(j, k) == static_cast<int>(k + 2));
}

TEST_CASE("block_rightCols_template", "[matrix][block]") {
    // This is the codepath that had the bug (operands swapped in offset calc)
    zipper::Matrix<int, 3, 6> M;
    for (zipper::index_type j = 0; j < 3; ++j)
        for (zipper::index_type k = 0; k < 6; ++k)
            M(j, k) = static_cast<int>(k);

    auto R = M.template rightCols<4>();
    STATIC_CHECK(std::decay_t<decltype(R)>::extents_type::static_extent(1) == 4);
    REQUIRE(R.extent(0) == 3);
    REQUIRE(R.extent(1) == 4);
    // Right 4 cols of a 6-col matrix: cols 2,3,4,5
    for (zipper::index_type j = 0; j < 3; ++j)
        for (zipper::index_type k = 0; k < 4; ++k)
            CHECK(R(j, k) == static_cast<int>(k + 2));
}

TEST_CASE("block_rightCols_template_all_cols", "[matrix][block]") {
    // Edge case: rightCols<N> where N == total cols
    zipper::Matrix<int, 2, 3> M;
    M(0, 0) = 10;
    M(0, 1) = 20;
    M(0, 2) = 30;
    M(1, 0) = 40;
    M(1, 1) = 50;
    M(1, 2) = 60;

    auto R = M.template rightCols<3>();
    REQUIRE(R.extent(1) == 3);
    CHECK(R(0, 0) == 10);
    CHECK(R(0, 1) == 20);
    CHECK(R(0, 2) == 30);
}

TEST_CASE("block_rightCols_template_one_col", "[matrix][block]") {
    // Edge case: rightCols<1>
    zipper::Matrix<int, 3, 4> M;
    for (zipper::index_type j = 0; j < 3; ++j)
        for (zipper::index_type k = 0; k < 4; ++k)
            M(j, k) = static_cast<int>(j * 10 + k);

    auto R = M.template rightCols<1>();
    REQUIRE(R.extent(0) == 3);
    REQUIRE(R.extent(1) == 1);
    CHECK(R(0, 0) == 3);
    CHECK(R(1, 0) == 13);
    CHECK(R(2, 0) == 23);
}

TEST_CASE("block_dynamic_matrix_template_blocks", "[matrix][block]") {
    // Test template-form blocks on a fully dynamic matrix
    zipper::Matrix<int, std::dynamic_extent, std::dynamic_extent> M(6, 4);
    for (zipper::index_type j = 0; j < 6; ++j)
        for (zipper::index_type k = 0; k < 4; ++k)
            M(j, k) = static_cast<int>(j * 10 + k);

    // topRows<2>
    auto T = M.template topRows<2>();
    REQUIRE(T.extent(0) == 2);
    REQUIRE(T.extent(1) == 4);
    CHECK(T(0, 0) == 0);
    CHECK(T(1, 3) == 13);

    // bottomRows<2>
    auto B = M.template bottomRows<2>();
    REQUIRE(B.extent(0) == 2);
    REQUIRE(B.extent(1) == 4);
    CHECK(B(0, 0) == 40);// row 4
    CHECK(B(1, 3) == 53);// row 5

    // leftCols<2>
    auto L = M.template leftCols<2>();
    REQUIRE(L.extent(0) == 6);
    REQUIRE(L.extent(1) == 2);
    CHECK(L(0, 0) == 0);
    CHECK(L(5, 1) == 51);

    // rightCols<2>
    auto R = M.template rightCols<2>();
    REQUIRE(R.extent(0) == 6);
    REQUIRE(R.extent(1) == 2);
    CHECK(R(0, 0) == 2);// col 2
    CHECK(R(5, 1) == 53);// col 3, row 5
}

TEST_CASE("block_write_through", "[matrix][block][mutable]") {
    // Verify that block views are mutable references to the original
    zipper::Matrix<int, 4, 4> M;
    for (zipper::index_type j = 0; j < 4; ++j)
        for (zipper::index_type k = 0; k < 4; ++k)
            M(j, k) = 0;

    // Write through bottomRows<2>
    auto B = M.template bottomRows<2>();
    B(0, 0) = 99;// row 2 of original
    CHECK(M(2, 0) == 99);

    // Write through rightCols<2>
    auto R = M.template rightCols<2>();
    R(0, 0) = 77;// col 2 of original
    CHECK(M(0, 2) == 77);

    // Write through topRows<2>
    auto T = M.template topRows<2>();
    T(1, 3) = 55;// row 1, col 3 of original
    CHECK(M(1, 3) == 55);

    // Write through leftCols<2>
    auto L = M.template leftCols<2>();
    L(3, 1) = 33;// row 3, col 1 of original
    CHECK(M(3, 1) == 33);
}

TEST_CASE("test_span_view", "[vector][storage][dense][span]") {
    using zipper::VectorBase;
    {
        zipper::Vector<double, 3> x = { 1, 2, 3 };

        VectorBase y = x.expression().as_span();
        CHECK(x == y);
        VectorBase z = x.expression().as_std_span();
        CHECK(x == z);
    }
    {
        zipper::Vector<double, std::dynamic_extent> x = { 1, 2, 3 };

        STATIC_CHECK(std::decay_t<decltype(x)>::extents_type::rank() == 1);

        VectorBase y = x.expression().as_span();
        CHECK(x == y);
        VectorBase z = x.expression().as_std_span();
        CHECK(x == z);
    }
}
