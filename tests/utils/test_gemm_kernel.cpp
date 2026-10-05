#include <array>
#include <limits>
#include <tuple>
#include <vector>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/utils/gemm.hpp>

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

TEST_CASE("gemm kernel: square sizes across tile/block edges",
          "[gemm][matrix][dense]") {
    // Spans sub-tile sizes, multiple mc/kc blocks, and non-multiples of the
    // MR/NR/KC block sizes to exercise ragged-edge handling.
    for (index_type n : {1, 2, 3, 5, 7, 8, 15, 16, 63, 64, 65, 100, 129, 200, 300})
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
    // Both layouts are eligible targets.
    STATIC_CHECK(gd::GemmEligible<DMat::expression_type, DMat::expression_type,
                                  DMatCol::expression_type>);

    // Exercise small sizes and ragged edges.
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

TEST_CASE("gemm kernel: writable non-contiguous sub-block target (via temporary)",
          "[gemm][matrix][dense][slice]") {
    namespace gd = zipper::expression::binary::detail;

    // A writable rank-2 sub-block view of a larger matrix is the canonical
    // non-contiguous target: a Slice (layout_stride, no data()). The gate must
    // admit it so it gets the kernel instead of the generic coefficient path.
    using BlockExpr = std::decay_t<decltype(std::declval<DMat&>().slice(
        zipper::slice(index_type{1}, index_type{2}),
        zipper::slice(index_type{1}, index_type{2})))>::expression_type;
    STATIC_CHECK(gd::WritableDenseRank2Target<BlockExpr>);
    STATIC_CHECK(gd::GemmEligible<DMat::expression_type, DMat::expression_type,
                                  BlockExpr>);

    // Cover a small size and a multi-tile size. For each,
    // place an M×N product into the interior of a larger (M+2pad)×(N+2pad)
    // matrix, leaving a border that must remain untouched.
    auto run = [](index_type M, index_type K, index_type N, index_type pad,
                  double sentinel) {
        DMat A = make_filled(M, K, 11), B = make_filled(K, N, 12);
        DMat big(M + 2 * pad, N + 2 * pad);
        for (index_type i = 0; i < big.extent(0); ++i)
            for (index_type j = 0; j < big.extent(1); ++j) big(i, j) = sentinel;

        auto block = big.slice(zipper::slice(pad, M), zipper::slice(pad, N));
        block = A * B;  // computed into a temporary, then assigned

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

    run(8, 8, 8, 2, 7.5);        // small
    run(100, 70, 90, 3, -3.25);  // multi-tile
    run(65, 1, 65, 1, 1.0);      // ragged edges
}

TEST_CASE("gemm kernel: in-place aliasing A = A * B stays correct",
          "[gemm][matrix][dense][aliasing]") {
    for (index_type n : {4, 64, 100}) {
        DMat A = make_filled(n, n, 3), B = make_filled(n, n, 4);
        DMat A_orig = A;
        A = A * B;  // aliases an input -> AssignHelper evaluates via a temp
        check_against_reference(A_orig, B, A);
    }
}

TEST_CASE("gemm kernel: aliasing through views stays correct",
          "[gemm][matrix][dense][aliasing]") {
    // The operand is a view of the target, so AssignHelper must still
    // evaluate into a temporary.
    const index_type n = 70;
    DMat C = make_filled(n, n, 7), B = make_filled(n, n, 8);
    DMat C_orig = C;
    C = C.slice(zipper::slice(0, n), zipper::slice(0, n)) * B;
    check_against_reference(C_orig, B, C);

    DMat D = make_filled(n, n, 9);
    DMat D_orig = D;
    D = D.transpose() * B;
    DMat Dt(n, n);
    for (index_type i = 0; i < n; ++i)
        for (index_type j = 0; j < n; ++j) Dt(i, j) = D_orig(j, i);
    check_against_reference(Dt, B, D);
}

TEST_CASE("gemm kernel: noalias() writes the target directly",
          "[gemm][matrix][dense][aliasing]") {
    for (index_type n : {5, 70, 130}) {
        DMat A = make_filled(n, n + 1, 10), B = make_filled(n + 1, n, 11);
        DMat C(n, n);
        DMatCol Ccol(n, n);
        C.noalias() = A * B;
        Ccol.noalias() = A * B;
        check_against_reference(A, B, C);
        check_against_reference(A, B, Ccol);
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

TEST_CASE("gemm kernel: non-default register tile shapes",
          "[gemm][matrix][dense]") {
    namespace g = zipper::expression::binary::detail::gemm;
    auto check_tile = [](auto tile_tag) {
        constexpr g::Tile S = decltype(tile_tag)::value;
        for (index_type n : {1, 5, 13, 70, 131}) {
            DMat A = make_filled(n, n + 3, 1), B = make_filled(n + 3, n, 2);
            DMat C(n, n);
            g::gemm_blocked<double, S>(A, B, C);
            check_against_reference(A, B, C);
        }
    };
    // mr must be a multiple of the native SIMD width (2, 4 or 8 doubles).
    constexpr auto W = g::simd_width<double>;
    check_tile(std::integral_constant<g::Tile, g::Tile{8, 6}>{});
    check_tile(std::integral_constant<g::Tile, g::Tile{8, 4}>{});
    check_tile(std::integral_constant<g::Tile, g::Tile{2 * W, 3}>{});
    check_tile(std::integral_constant<g::Tile, g::default_tile<double>>{});
}

TEST_CASE("gemm kernel: five-loop blocking with small block sizes",
          "[gemm][matrix][dense][blocking]") {
    namespace g = zipper::expression::binary::detail::gemm;
    constexpr g::Tile S = g::default_tile<double>;
    // Blocks much smaller than the matrices: many jc / pc / ic iterations,
    // ragged blocks at every level (nc, mc not multiples of the tile).
    for (g::BlockSizes b : {g::BlockSizes{.kc = 8, .mc = S.mr, .nc = S.nr},
                            g::BlockSizes{.kc = 13, .mc = 3 * S.mr + 1,
                                          .nc = 2 * S.nr + 1}}) {
        g::GemmConfig<double, S> config{.blocks = b};
        for (auto [M, K, N] : {std::array<index_type, 3>{37, 29, 41},
                               std::array<index_type, 3>{64, 64, 64},
                               std::array<index_type, 3>{5, 100, 3}}) {
            DMat A = make_filled(M, K, 12), B = make_filled(K, N, 13);
            DMat C(M, N);
            g::gemm_blocked<double, S>(A, B, C, config);
            check_against_reference(A, B, C);
        }
    }
}

TEST_CASE("gemm kernel: blocking model", "[gemm][blocking]") {
    namespace g = zipper::expression::binary::detail::gemm;
    constexpr g::Tile S{.mr = 8, .nr = 6};
    constexpr auto b = g::blocking_for<double, S>(
        g::CacheSizes{.l1d = 32 * 1024, .l2 = 512 * 1024, .l3 = 4 << 20});
    // L1/2 / ((mr + nr) * 8 B) = 146 -> 144; L2/2 / (kc * 8 B) = 227 -> 224.
    STATIC_CHECK(b.kc == 144);
    STATIC_CHECK(b.mc == 224);
    STATIC_CHECK(b.mc % S.mr == 0);
    STATIC_CHECK(b.nc % S.nr == 0);
    // Tiny caches still give at least one tile per block.
    constexpr auto t = g::blocking_for<double, S>(g::CacheSizes{1, 1, 1});
    STATIC_CHECK(t.kc >= 1);
    STATIC_CHECK(t.mc >= S.mr);
    STATIC_CHECK(t.nc >= S.nr);
}

// ── alpha / beta (BLAS gemm) ────────────────────────────────────────────

namespace {
// Reference: beta * C0 + alpha * A * B, never reading C0 when beta == 0.
DMat reference_gemm(double alpha, const DMat& A, const DMat& B, double beta,
                    const DMat& C0) {
    const index_type M = A.extent(0), K = A.extent(1), N = B.extent(1);
    DMat R(M, N);
    for (index_type i = 0; i < M; ++i)
        for (index_type j = 0; j < N; ++j) {
            double ab = 0.0;
            for (index_type k = 0; k < K; ++k) ab += A(i, k) * B(k, j);
            R(i, j) = (beta == 0.0 ? 0.0 : beta * C0(i, j)) + alpha * ab;
        }
    return R;
}

template <typename CMat>
void check_equal(const DMat& R, const CMat& C) {
    REQUIRE(C.extent(0) == R.extent(0));
    REQUIRE(C.extent(1) == R.extent(1));
    for (index_type i = 0; i < R.extent(0); ++i)
        for (index_type j = 0; j < R.extent(1); ++j)
            CHECK(C(i, j) == Catch::Approx(R(i, j)).epsilon(1e-12).margin(1e-12));
}

template <typename S>
double as_double(S s) {
    return static_cast<double>(s);
}
}  // namespace

TEST_CASE("gemm: every static/run-time alpha and beta combination",
          "[gemm][matrix][dense][blas]") {
    using zipper::cw;
    namespace g = zipper::expression::binary::detail::gemm;
    auto run = [](auto alpha, auto beta) {
        for (auto [M, K, N] : {std::array<index_type, 3>{7, 5, 9},
                               std::array<index_type, 3>{70, 130, 33}}) {
            DMat A = make_filled(M, K, 20), B = make_filled(K, N, 21);
            DMat C0 = make_filled(M, N, 22);
            DMat C = C0;
            zipper::utils::gemm(alpha, A, B, beta, C);
            check_equal(reference_gemm(as_double(alpha), A, B, as_double(beta), C0), C);

            // Same through the kernel entry with small blocks (multi-pc: beta
            // must be applied by the first kc block only).
            DMat D = C0;
            g::GemmConfig<double, g::default_tile<double>> cfg{
                .blocks = {.kc = 16, .mc = 16, .nc = 12}};
            g::gemm_blocked<double>(alpha, A, B, beta, D, cfg);
            check_equal(reference_gemm(as_double(alpha), A, B, as_double(beta), C0), D);
        }
    };
    auto alphas = std::tuple{cw<1>, cw<-1>, cw<0>,
                             2.5, 1.0, 0.0, -1.0};
    auto betas = std::tuple{cw<0>, cw<1>, cw<-1>,
                            0.5, 0.0, 1.0};
    std::apply([&](auto... a) {
        ([&](auto alpha) {
            std::apply([&](auto... b) { (run(alpha, b), ...); }, betas);
        }(a), ...);
    }, alphas);
}

TEST_CASE("gemm: beta = 0 never reads C (NaN does not leak)",
          "[gemm][matrix][dense][blas]") {
    using zipper::cw;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (index_type n : {5, 70}) {
        DMat A = make_filled(n, n, 23), B = make_filled(n, n, 24);
        DMat Cnan(n, n);
        for (index_type i = 0; i < n; ++i)
            for (index_type j = 0; j < n; ++j) Cnan(i, j) = nan;
        const DMat R = reference_gemm(1.0, A, B, 0.0, Cnan);

        DMat C = Cnan;
        zipper::utils::gemm(cw<1>, A, B, cw<0>, C);
        check_equal(R, C);

        DMat D = Cnan;
        zipper::utils::gemm(1.0, A, B, 0.0, D);  // run-time zero
        check_equal(R, D);

        DMat E = Cnan;
        E.noalias() = A * B;
        check_equal(R, E);
    }
}

TEST_CASE("gemm: empty inner dimension scales C by beta",
          "[gemm][matrix][dense][blas]") {
    DMat A(4, 0), B(0, 3);
    DMat C0 = make_filled(4, 3, 25);
    DMat C = C0;
    zipper::utils::gemm(2.0, A, B, 0.5, C);
    for (index_type i = 0; i < 4; ++i)
        for (index_type j = 0; j < 3; ++j) CHECK(C(i, j) == 0.5 * C0(i, j));
}

TEST_CASE("gemm: scaled products reach the kernel via the accumulate protocol",
          "[gemm][matrix][dense][blas]") {
    namespace ed = zipper::expression::detail;
    const index_type n = 70;
    DMat A = make_filled(n, n + 3, 26), B = make_filled(n + 3, n, 27);
    const DMat C0 = make_filled(n, n, 28);

    STATIC_CHECK(ed::HasAccumulateStrategy<decltype((A * B).expression())>);
    STATIC_CHECK(ed::HasAccumulateStrategy<decltype((2.0 * (A * B)).expression())>);
    STATIC_CHECK(ed::HasAccumulateStrategy<decltype(((A * B) * 2.0).expression())>);
    STATIC_CHECK(ed::HasAccumulateStrategy<decltype(((A * B) / 2.0).expression())>);
    STATIC_CHECK(ed::HasAccumulateStrategy<decltype((-(A * B)).expression())>);
    STATIC_CHECK(ed::HasAccumulateStrategy<
                 decltype((-(3.0 * (A * B) / 2.0)).expression())>);

    // Only linear scalings forward: s / X, s + X, X - s do not.
    namespace ud = zipper::expression::unary::detail;
    STATIC_CHECK(ud::is_linear_scaling_v<std::multiplies<double>, true>);
    STATIC_CHECK(ud::is_linear_scaling_v<std::multiplies<double>, false>);
    STATIC_CHECK(ud::is_linear_scaling_v<std::divides<double>, true>);
    STATIC_CHECK_FALSE(ud::is_linear_scaling_v<std::divides<double>, false>);
    STATIC_CHECK_FALSE(ud::is_linear_scaling_v<std::plus<double>, true>);
    STATIC_CHECK_FALSE(ud::is_linear_scaling_v<std::minus<double>, true>);

    DMat C(n, n);
    C = 2.0 * (A * B);
    check_equal(reference_gemm(2.0, A, B, 0.0, C0), C);
    C = (A * B) / 4.0;
    check_equal(reference_gemm(0.25, A, B, 0.0, C0), C);
    C = -(A * B);
    check_equal(reference_gemm(-1.0, A, B, 0.0, C0), C);
    C = -(3.0 * (A * B) / 2.0);
    check_equal(reference_gemm(-1.5, A, B, 0.0, C0), C);
    DMatCol Ccol(n, n);
    Ccol = 2.0 * (A * B);
    check_equal(reference_gemm(2.0, A, B, 0.0, C0), Ccol);
}

TEST_CASE("gemm: compound assignment entry points",
          "[gemm][matrix][dense][blas]") {
    const index_type n = 70;
    DMat A = make_filled(n, n + 3, 29), B = make_filled(n + 3, n, 30);
    const DMat C0 = make_filled(n, n, 31);

    DMat C = C0;
    C += A * B;
    check_equal(reference_gemm(1.0, A, B, 1.0, C0), C);
    C = C0;
    C -= A * B;
    check_equal(reference_gemm(-1.0, A, B, 1.0, C0), C);
    C = C0;
    C += 2.0 * (A * B);
    check_equal(reference_gemm(2.0, A, B, 1.0, C0), C);

    C = C0;
    C.noalias() += A * B;
    check_equal(reference_gemm(1.0, A, B, 1.0, C0), C);
    C = C0;
    C.noalias() -= A * B;
    check_equal(reference_gemm(-1.0, A, B, 1.0, C0), C);
    C = C0;
    C.noalias() += -(A * B) / 2.0;
    check_equal(reference_gemm(-0.5, A, B, 1.0, C0), C);
    // noalias += of a non-accumulable expression: pointwise update.
    C = C0;
    C.noalias() += A.slice(zipper::slice(0, n), zipper::slice(0, n));
    for (index_type i = 0; i < n; ++i)
        for (index_type j = 0; j < n; ++j) CHECK(C(i, j) == C0(i, j) + A(i, j));

    // Aliasing compound assignment: C += C * Bsq reads C while writing it.
    DMat Bsq = make_filled(n, n, 32);
    C = C0;
    C += C * Bsq;
    check_equal(reference_gemm(1.0, C0, Bsq, 1.0, C0), C);

    // Strided sub-block target.
    DMat Big = make_filled(n + 4, n + 4, 33);
    const DMat Big0 = Big;
    auto block = Big.slice(zipper::slice(2, n), zipper::slice(2, n));
    block.noalias() += A * B;
    for (index_type i = 0; i < n + 4; ++i)
        for (index_type j = 0; j < n + 4; ++j) {
            const bool inside = i >= 2 && i < n + 2 && j >= 2 && j < n + 2;
            if (!inside) CHECK(Big(i, j) == Big0(i, j));
        }
    DMat inner(n, n);
    for (index_type i = 0; i < n; ++i)
        for (index_type j = 0; j < n; ++j) inner(i, j) = Big0(i + 2, j + 2);
    DMat got(n, n);
    for (index_type i = 0; i < n; ++i)
        for (index_type j = 0; j < n; ++j) got(i, j) = Big(i + 2, j + 2);
    check_equal(reference_gemm(1.0, A, B, 1.0, inner), got);
}

TEST_CASE("gemm: small static matrices use the generic fallback",
          "[gemm][matrix][blas]") {
    using zipper::cw;
    Matrix<double, 3, 3> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};
    Matrix<double, 3, 3> B{{2, 0, 1}, {1, 3, 0}, {0, 1, 4}};
    Matrix<double, 3, 3> C{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    Matrix<double, 3, 3> AB = A * B;
    zipper::utils::gemm(2.0, A, B, cw<1>, C);
    for (index_type i = 0; i < 3; ++i)
        for (index_type j = 0; j < 3; ++j) CHECK(C(i, j) == 1.0 + 2.0 * AB(i, j));
}

#if defined(ZIPPER_TEST_HAVE_TBB)
TEST_CASE("gemm: parallel execution policy", "[gemm][matrix][dense][parallel]") {
    using zipper::cw;
    namespace g = zipper::expression::binary::detail::gemm;
    for (auto [M, K, N] : {std::array<index_type, 3>{1, 7, 5},
                           std::array<index_type, 3>{37, 29, 41},
                           std::array<index_type, 3>{300, 200, 150},
                           std::array<index_type, 3>{513, 64, 3}}) {
        DMat A = make_filled(M, K, 34), B = make_filled(K, N, 35);
        const DMat C0 = make_filled(M, N, 36);

        DMat C = C0;
        zipper::utils::gemm(std::execution::par, 0.5, A, B, cw<1>, C);
        check_equal(reference_gemm(0.5, A, B, 1.0, C0), C);

        DMatCol Ccol(M, N);
        zipper::utils::gemm(std::execution::par, cw<1>, A, B,
                            cw<0>, Ccol);
        check_equal(reference_gemm(1.0, A, B, 0.0, C0), Ccol);

        // Small blocks: many parallel row blocks per (jc, pc), multi-pc beta.
        DMat D = C0;
        constexpr g::Tile S = g::default_tile<double>;
        g::GemmConfig<double, S, std::execution::parallel_policy> cfg{
            .blocks = {.kc = 16, .mc = S.mr, .nc = 4 * S.nr}};
        g::gemm_blocked<double, S>(-1.0, A, B, 2.0, D, cfg);
        check_equal(reference_gemm(-1.0, A, B, 2.0, C0), D);
    }
}
#endif

// ── Dimensioning (gemm_config.hpp) and float ────────────────────────────

TEST_CASE("gemm tuning: default tiles fit the register budget",
          "[gemm][tuning]") {
    namespace g = zipper::expression::binary::detail::gemm;
    using TF = g::GemmTuning<float>;
    using TD = g::GemmTuning<double>;
    // Width derives from one vector size: float lanes are twice double's.
    STATIC_CHECK(TF::vector_bytes == TD::vector_bytes);
    STATIC_CHECK(TF::simd_width == 2 * TD::simd_width);
    // Same shape in vectors, so the same register use for both types.
    STATIC_CHECK(g::default_tile<float>.mr == 2 * g::default_tile<double>.mr);
    STATIC_CHECK(g::default_tile<float>.nr == g::default_tile<double>.nr);
    STATIC_CHECK(g::live_registers(g::default_tile<float>, TF::simd_width)
                 <= TF::registers);
    STATIC_CHECK(g::live_registers(g::default_tile<double>, TD::simd_width)
                 <= TD::registers);
    // The budget formula: 16 registers -> 2 x 6, 32 -> 2 x 14.
    STATIC_CHECK(g::max_nr(16, 2) == 6);
    STATIC_CHECK(g::max_nr(32, 2) == 14);
    STATIC_CHECK(g::live_registers(g::Tile{.mr = 8, .nr = 6}, 4) == 15);
    STATIC_CHECK(g::live_registers(g::Tile{.mr = 8, .nr = 8}, 4) == 19);  // spills
}

TEST_CASE("gemm kernel: float", "[gemm][matrix][dense][float]") {
    using FMat = Matrix<float, dynamic_extent, dynamic_extent>;
    using FMatCol = Matrix<float, dynamic_extent, dynamic_extent, false>;
    auto to_float = [](const DMat& D) {
        FMat F(D.extent(0), D.extent(1));
        for (index_type i = 0; i < D.extent(0); ++i)
            for (index_type j = 0; j < D.extent(1); ++j)
                F(i, j) = static_cast<float>(D(i, j));
        return F;
    };
    for (auto [M, K, N] : {std::array<index_type, 3>{1, 1, 1},
                           std::array<index_type, 3>{7, 5, 9},
                           std::array<index_type, 3>{37, 29, 41},
                           std::array<index_type, 3>{130, 70, 200}}) {
        const DMat Ad = make_filled(M, K, 40), Bd = make_filled(K, N, 41);
        const FMat A = to_float(Ad), B = to_float(Bd);
        FMat C = A * B;
        FMatCol Ccol = 0.5f * (A * B);
        for (index_type i = 0; i < M; ++i)
            for (index_type j = 0; j < N; ++j) {
                double ref = 0.0;
                for (index_type k = 0; k < K; ++k)
                    ref += double(A(i, k)) * double(B(k, j));
                CHECK(C(i, j) == Catch::Approx(ref).epsilon(1e-5).margin(1e-4));
                CHECK(Ccol(i, j)
                      == Catch::Approx(0.5 * ref).epsilon(1e-5).margin(1e-4));
            }
    }
}
