#include <vector>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>

#include "../catch_include.hpp"

using namespace zipper;

namespace {

using DMat = Matrix<double, dynamic_extent, dynamic_extent>;

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

// Independent reference: plain triple loop, no expression templates.
void check_against_reference(const DMat& A, const DMat& B, const DMat& C) {
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
