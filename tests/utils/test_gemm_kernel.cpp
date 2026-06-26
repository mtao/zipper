#include <vector>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/binary/detail/gemm_eligible.hpp>

#include "../catch_include.hpp"

using namespace zipper;

namespace {

using DMat = Matrix<double, dynamic_extent, dynamic_extent>;
using DMatCol = Matrix<double, dynamic_extent, dynamic_extent, false>;  // col-major

// Deterministic fill, distinct per (salt) so A != B.
DMat make_filled(index_type rows, index_type cols, unsigned salt) {
    DMat M(rows, cols);
    unsigned s = salt * 2654435761u + 1u;
    for (index_type i = 0; i < rows; ++i)
        for (index_type j = 0; j < cols; ++j) {
            s = s * 1103515245u + 12345u;
            M(i, j) = static_cast<double>((s >> 9) % 1000) / 500.0 - 1.0;
        }
    return M;
}

// Independent reference: plain triple loop, no expression templates. Templated
// on the target type so it also validates column-major results.
template <typename CMat>
void check_against_reference(const DMat& A, const DMat& B, const CMat& C) {
    const index_type M = A.extent(0), K = A.extent(1), N = B.extent(1);
    REQUIRE(C.extent(0) == M);
    REQUIRE(C.extent(1) == N);
    for (index_type i = 0; i < M; ++i)
        for (index_type j = 0; j < N; ++j) {
            double ref = 0.0;
            for (index_type k = 0; k < K; ++k) ref += A(i, k) * B(k, j);
            CHECK(C(i, j) == Catch::Approx(ref).epsilon(1e-12).margin(1e-12));
        }
}

void check_product(index_type M, index_type K, index_type N) {
    DMat A = make_filled(M, K, 1), B = make_filled(K, N, 2);
    DMat C = A * B;  // routed through the blocked kernel for eligible operands
    check_against_reference(A, B, C);
}

}  // namespace

TEST_CASE("gemm kernel: square sizes across tier/block edges",
          "[gemm][matrix][dense]") {
    // Spans ikj tier (<64), blocked tier (>=64), and non-multiples of the
    // MR/NR/KC block sizes to exercise ragged-edge handling.
    for (index_type n : {1, 2, 3, 5, 7, 8, 15, 16, 63, 64, 65, 100, 129, 200})
        check_product(n, n, n);
}

TEST_CASE("gemm kernel: rectangular M != K != N", "[gemm][matrix][dense]") {
    check_product(1, 7, 1);
    check_product(3, 64, 5);
    check_product(65, 1, 65);
    check_product(70, 130, 33);
    check_product(129, 65, 200);
    check_product(200, 33, 129);
}

TEST_CASE("gemm kernel: column-major target routes through the kernel",
          "[gemm][matrix][dense][layout]") {
    namespace gd = zipper::expression::binary::detail;
    // Both layouts are eligible targets — the gate is no longer row-major-only.
    STATIC_CHECK(gd::DenseContiguousTarget<DMat::expression_type>);
    STATIC_CHECK(gd::DenseContiguousTarget<DMatCol::expression_type>);

    // Column-major C(M×N) is written via the transpose identity Cᵀ = Bᵀ·Aᵀ;
    // exercise the ikj tier, the blocked tier, and ragged edges.
    for (index_type n : {3, 8, 64, 100, 129}) {
        DMat A = make_filled(n, n, 1), B = make_filled(n, n, 2);
        DMatCol C = A * B;
        check_against_reference(A, B, C);
    }
    // Rectangular, blocked: M != K != N.
    {
        DMat A = make_filled(70, 130, 5), B = make_filled(130, 33, 6);
        DMatCol C = A * B;
        check_against_reference(A, B, C);
    }
}

TEST_CASE("gemm kernel: writable non-contiguous sub-block target (PATH 2)",
          "[gemm][matrix][dense][slice]") {
    namespace gd = zipper::expression::binary::detail;

    // A writable rank-2 sub-block view of a larger matrix is the canonical
    // PATH-2 target: it is a Slice (layout_stride, no data()), so it is NOT a
    // DenseContiguousTarget, yet it is a writable dense rank-2 expression. The
    // gate must admit it via WritableDenseRank2Target so gemm() routes it to
    // the scatter path instead of the generic coefficient product.
    using BlockExpr = std::decay_t<decltype(std::declval<DMat&>().slice(
        zipper::slice(index_type{1}, index_type{2}),
        zipper::slice(index_type{1}, index_type{2})))>::expression_type;
    STATIC_CHECK(gd::WritableDenseRank2Target<BlockExpr>);
    STATIC_CHECK_FALSE(gd::DenseContiguousTarget<BlockExpr>);
    STATIC_CHECK(gd::GemmEligible<DMat::expression_type, DMat::expression_type,
                                  BlockExpr>);

    // Cover an ikj-tier size (<64) and a blocked-tier size (>=64). For each,
    // place an M×N product into the interior of a larger (M+2pad)×(N+2pad)
    // matrix, leaving a border that must remain untouched.
    auto run = [](index_type M, index_type K, index_type N, index_type pad,
                  double sentinel) {
        DMat A = make_filled(M, K, 11), B = make_filled(K, N, 12);
        DMat big(M + 2 * pad, N + 2 * pad);
        for (index_type i = 0; i < big.extent(0); ++i)
            for (index_type j = 0; j < big.extent(1); ++j) big(i, j) = sentinel;

        auto block = big.slice(zipper::slice(pad, M), zipper::slice(pad, N));
        block = A * B;  // routed through PATH 2 (scatter into the strided view)

        // (a) the block equals the independent triple-loop reference
        check_against_reference(A, B, block);

        // (b) every element outside the block is UNTOUCHED
        for (index_type i = 0; i < big.extent(0); ++i)
            for (index_type j = 0; j < big.extent(1); ++j) {
                const bool inside = (i >= pad && i < pad + M && j >= pad &&
                                     j < pad + N);
                if (!inside) CHECK(big(i, j) == sentinel);
            }
    };

    run(8, 8, 8, 2, 7.5);        // ikj tier
    run(100, 70, 90, 3, -3.25);  // blocked tier
    run(65, 1, 65, 1, 1.0);      // blocked tier, ragged edges
}

TEST_CASE("gemm kernel: in-place aliasing A = A * B stays correct",
          "[gemm][matrix][dense][aliasing]") {
    for (index_type n : {4, 64, 100}) {
        DMat A = make_filled(n, n, 3), B = make_filled(n, n, 4);
        DMat A_orig = A;
        A = A * B;  // output base aliases an input -> kernel must use a temp
        check_against_reference(A_orig, B, A);
    }
}

TEST_CASE("gemm kernel: non-contiguous / lazy operands (expression-accessor "
          "packing path)",
          "[gemm][matrix][dense]") {
    // Since the kernel reads operands through their element accessor, these
    // dense-but-not-contiguous operands are now GEMM-eligible (packed via the
    // concept fallback path), not routed to the generic coeff product.
    SECTION("transposed operand") {
        DMat A = make_filled(40, 50, 5), Bt = make_filled(70, 50, 6);
        DMat C = A * Bt.transpose();  // A(40x50) * Bt^T(50x70)
        REQUIRE(C.extent(0) == 40);
        REQUIRE(C.extent(1) == 70);
        for (index_type i = 0; i < 40; ++i)
            for (index_type j = 0; j < 70; ++j) {
                double ref = 0.0;
                for (index_type k = 0; k < 50; ++k) ref += A(i, k) * Bt(j, k);
                CHECK(C(i, j) ==
                      Catch::Approx(ref).epsilon(1e-12).margin(1e-12));
            }
    }
    SECTION("lazy scaled operand (2*A)*B packs on the fly") {
        DMat A = make_filled(70, 65, 7), B = make_filled(65, 80, 8);
        DMat C = (2.0 * A) * B;
        for (index_type i = 0; i < 70; ++i)
            for (index_type j = 0; j < 80; ++j) {
                double ref = 0.0;
                for (index_type k = 0; k < 65; ++k)
                    ref += (2.0 * A(i, k)) * B(k, j);
                CHECK(C(i, j) ==
                      Catch::Approx(ref).epsilon(1e-12).margin(1e-12));
            }
    }
}

TEST_CASE("gemm kernel: ineligible operands use the generic path correctly",
          "[gemm][matrix][dense][fallback]") {
    SECTION("small static matrices use the generic unrolled path") {
        Matrix<double, 3, 3> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        Matrix<double, 3, 3> B{{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
        Matrix<double, 3, 3> C = A * B;
        for (index_type i = 0; i < 3; ++i)
            for (index_type j = 0; j < 3; ++j) {
                double ref = 0.0;
                for (index_type k = 0; k < 3; ++k) ref += A(i, k) * B(k, j);
                CHECK(C(i, j) == Catch::Approx(ref));
            }
    }
}

namespace {

// Fill a statically-sized matrix with the same deterministic pattern used for
// the dynamic operands above (so a static M×N matches its dynamic twin).
template <typename Mat>
void fill_static(Mat& M, index_type rows, index_type cols, unsigned salt) {
    unsigned s = salt * 2654435761u + 1u;
    for (index_type i = 0; i < rows; ++i)
        for (index_type j = 0; j < cols; ++j) {
            s = s * 1103515245u + 12345u;
            M(i, j) = static_cast<double>((s >> 9) % 1000) / 500.0 - 1.0;
        }
}

}  // namespace

// Lock in the gate decision itself (not just the numeric result): large static
// operands must be GemmEligible (routed to the kernel), small ones must not.
TEST_CASE("gemm gate: large static eligible, small static not",
          "[gemm][matrix][dense][static]") {
    namespace gd = zipper::expression::binary::detail;
    using Big = Matrix<double, 80, 80>::expression_type;
    using Small = Matrix<double, 3, 3>::expression_type;
    using Dyn = DMat::expression_type;
    using OneDimBig = Matrix<double, 64, 8>::expression_type;

    STATIC_CHECK(gd::GemmEligible<Big, Big, Big>);
    STATIC_CHECK(gd::GemmEligible<OneDimBig, OneDimBig, OneDimBig>);
    STATIC_CHECK(gd::GemmEligible<Dyn, Dyn, Dyn>);  // unchanged dynamic path
    STATIC_CHECK_FALSE(gd::GemmEligible<Small, Small, Small>);
    // Mixing a small static target with large static sources is still gated
    // out, since every operand must be size-admissible.
    STATIC_CHECK_FALSE(gd::GemmEligible<Big, Big, Small>);
}

// The gate (gemm_eligible.hpp) admits a fully-static operand once a dimension
// reaches gemm_static_dim_threshold (== 64), so these large static products are
// routed through the blocked kernel; correctness is the same triple-loop check.
TEST_CASE("gemm kernel: large static-extent operands route through the kernel",
          "[gemm][matrix][dense][static]") {
    SECTION("square, exact tile multiple (80x80)") {
        Matrix<double, 80, 80> A, B;
        fill_static(A, 80, 80, 11);
        fill_static(B, 80, 80, 12);
        Matrix<double, 80, 80> C = A * B;
        for (index_type i = 0; i < 80; ++i)
            for (index_type j = 0; j < 80; ++j) {
                double ref = 0.0;
                for (index_type k = 0; k < 80; ++k) ref += A(i, k) * B(k, j);
                CHECK(C(i, j) ==
                      Catch::Approx(ref).epsilon(1e-12).margin(1e-12));
            }
    }
    SECTION("rectangular, non-tile-multiple (100x72 = 100x96 * 96x72)") {
        Matrix<double, 100, 96> A;
        Matrix<double, 96, 72> B;
        fill_static(A, 100, 96, 13);
        fill_static(B, 96, 72, 14);
        Matrix<double, 100, 72> C = A * B;
        for (index_type i = 0; i < 100; ++i)
            for (index_type j = 0; j < 72; ++j) {
                double ref = 0.0;
                for (index_type k = 0; k < 96; ++k) ref += A(i, k) * B(k, j);
                CHECK(C(i, j) ==
                      Catch::Approx(ref).epsilon(1e-12).margin(1e-12));
            }
    }
    SECTION("large by a single dimension (64x8 * 8x64)") {
        // Only the 64 dimension reaches the threshold; still admitted.
        Matrix<double, 64, 8> A;
        Matrix<double, 8, 64> B;
        fill_static(A, 64, 8, 15);
        fill_static(B, 8, 64, 16);
        Matrix<double, 64, 64> C = A * B;
        for (index_type i = 0; i < 64; ++i)
            for (index_type j = 0; j < 64; ++j) {
                double ref = 0.0;
                for (index_type k = 0; k < 8; ++k) ref += A(i, k) * B(k, j);
                CHECK(C(i, j) ==
                      Catch::Approx(ref).epsilon(1e-12).margin(1e-12));
            }
    }
}
