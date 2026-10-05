// Tiled relayout (TiledRelayout.hpp): which assignments qualify, and that
// every assignment produces exactly what the coefficient path would. Also
// built as test_tiled_relayout_always with ZIPPER_RELAYOUT_TILING_MIN_BYTES=0,
// so every relayout here (small, ragged, rank 3) goes through the tiles.

#include <array>
#include <vector>

#include <zipper/Matrix.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/detail/TiledRelayout.hpp>

#include "../catch_include.hpp"

using namespace zipper;
using zipper::expression::detail::TiledRelayout;

namespace {
template <typename Z>
using expr_t = std::remove_cvref_t<decltype(std::declval<Z>().expression())>;

template <bool RowMajor>
auto filled(index_type r, index_type c) {
    Matrix<double, dynamic_extent, dynamic_extent, RowMajor> M(r, c);
    for (index_type i = 0; i < r; ++i)
        for (index_type j = 0; j < c; ++j) M(i, j) = 100.0 * i + j;
    return M;
}
} // namespace

TEST_CASE("tiled relayout: selection is compile-time", "[assign][relayout]") {
    using R = Matrix<double, dynamic_extent, dynamic_extent>;
    using C = Matrix<double, dynamic_extent, dynamic_extent, false>;
    R M(4, 4);
    using row = expr_t<R>;
    using col = expr_t<C>;
    using block = expr_t<decltype(M.slice(zipper::slice(0, 2), zipper::slice(1, 2)))>;
    using trans = expr_t<decltype(M.transpose())>;
    using column = expr_t<decltype(M.col(1))>;
    using gather = expr_t<decltype(M.slice(full_extent_t{}, std::vector<index_type>{1, 3}))>;
    using sum = expr_t<decltype(M + M)>;

    // Relayouts: the fastest dimensions differ.
    STATIC_CHECK(TiledRelayout<row, col>);
    STATIC_CHECK(TiledRelayout<col, row>);
    STATIC_CHECK(TiledRelayout<trans, row>);
    STATIC_CHECK(TiledRelayout<block, col>);
    STATIC_CHECK(TiledRelayout<expr_t<MatrixXX<float>>, col>);
    // Same fastest dimension: never tiled, no run-time check emitted.
    STATIC_CHECK_FALSE(TiledRelayout<row, row>);
    STATIC_CHECK_FALSE(TiledRelayout<block, row>);
    STATIC_CHECK_FALSE(TiledRelayout<trans, col>);
    // Not linear arrays, unknown fastest dimension, or rank 1.
    STATIC_CHECK_FALSE(TiledRelayout<gather, col>);
    STATIC_CHECK_FALSE(TiledRelayout<sum, col>);
    STATIC_CHECK_FALSE(TiledRelayout<column, expr_t<VectorX<double>>>);
    // Fully static: decided by size at compile time.
#if !defined(ZIPPER_RELAYOUT_TILING_MIN_BYTES)
    STATIC_CHECK_FALSE(TiledRelayout<expr_t<Matrix<double, 4, 4>>,
                                     expr_t<Matrix<double, 4, 4, false>>>);
    STATIC_CHECK(TiledRelayout<expr_t<Matrix<double, 256, 256>>,
                               expr_t<Matrix<double, 256, 256, false>>>);
#endif
}

TEMPLATE_TEST_CASE_SIG("tiled relayout: matches the coefficient path",
                       "[assign][relayout]",
                       ((bool SrcRM, bool DstRM), SrcRM, DstRM),
                       (true, true), (true, false), (false, true), (false, false)) {
    const auto M = filled<SrcRM>(7, 5);
    using Dst = Matrix<double, dynamic_extent, dynamic_extent, DstRM>;

    SECTION("whole matrix") {
        Dst D(7, 5);
        D = M;
        for (index_type i = 0; i < 7; ++i)
            for (index_type j = 0; j < 5; ++j) CHECK(D(i, j) == M(i, j));
    }
    SECTION("transpose") {
        Dst D(5, 7);
        D = M.transpose();
        for (index_type i = 0; i < 5; ++i)
            for (index_type j = 0; j < 7; ++j) CHECK(D(i, j) == M(j, i));
    }
    SECTION("block of a block into a block") {
        Dst D = filled<DstRM>(9, 9);
        const Dst D0 = D;
        auto src = M.slice(zipper::slice(1, 5), zipper::slice(1, 4))
                       .slice(zipper::slice(1, 3), zipper::slice(0, 3));
        D.slice(zipper::slice(4, 3), zipper::slice(2, 3)) = src;
        for (index_type i = 0; i < 9; ++i)
            for (index_type j = 0; j < 9; ++j) {
                const bool in = i >= 4 && i < 7 && j >= 2 && j < 5;
                CHECK(D(i, j) == (in ? M(2 + i - 4, 1 + j - 2) : D0(i, j)));
            }
    }
    SECTION("strided slices") {
        Dst D(3, 2);
        D = M.slice(strided_slice{0, 7, 3}, strided_slice{1, 4, 2});
        for (index_type i = 0; i < 3; ++i)
            for (index_type j = 0; j < 2; ++j) CHECK(D(i, j) == M(3 * i, 1 + 2 * j));
    }
    SECTION("rows and columns") {
        VectorX<double> v(7);
        v = M.col(3);
        for (index_type i = 0; i < 7; ++i) CHECK(v(i) == M(i, 3));
        Dst D = filled<DstRM>(7, 5);
        D.row(2) = M.row(6);
        D.col(0) = M.col(4);
        for (index_type j = 1; j < 5; ++j) CHECK(D(2, j) == M(6, j));
        for (index_type i = 0; i < 7; ++i) CHECK(D(i, 0) == M(i, 4));
    }
    SECTION("gather through a transpose (coefficient path)") {
        Dst D(2, 7);
        D = M.slice(full_extent_t{}, std::vector<index_type>{4, 1}).transpose();
        for (index_type i = 0; i < 7; ++i) {
            CHECK(D(0, i) == M(i, 4));
            CHECK(D(1, i) == M(i, 1));
        }
    }
    SECTION("static extents and conversion") {
        Matrix<float, 3, 2, SrcRM> F;
        for (index_type i = 0; i < 3; ++i)
            for (index_type j = 0; j < 2; ++j) F(i, j) = 0.5f * float(i + 3 * j);
        Matrix<double, 3, 2, DstRM> D;
        D = F;
        for (index_type i = 0; i < 3; ++i)
            for (index_type j = 0; j < 2; ++j) CHECK(D(i, j) == double(F(i, j)));
    }
}

// ── rank 3 and the tiled relayout ───────────────────────────────────────

namespace {
template <bool LeftMajor>
auto filled3(index_type a, index_type b, index_type c) {
    Tensor_<double, zipper::dextents<3>, LeftMajor> T(a, b, c);
    for (index_type i = 0; i < a; ++i)
        for (index_type j = 0; j < b; ++j)
            for (index_type k = 0; k < c; ++k) T(i, j, k) = 1e4 * i + 1e2 * j + k;
    return T;
}
} // namespace

TEMPLATE_TEST_CASE_SIG("tiled relayout: rank 3", "[assign][relayout][tensor]",
                       ((bool SrcLM, bool DstLM), SrcLM, DstLM),
                       (true, true), (true, false), (false, true), (false, false)) {
    using Dst = Tensor_<double, zipper::dextents<3>, DstLM>;
    // Small (runs only) and large (> 256 KiB: tiled, with ragged tiles).
    for (auto [a, b, c] : {std::array<index_type, 3>{3, 4, 5},
                           std::array<index_type, 3>{40, 30, 23}}) {
        const auto T = filled3<SrcLM>(a, b, c);
        SECTION("copy") {
            Dst D(a, b, c);
            D = T;
            for (index_type i = 0; i < a; ++i)
                for (index_type j = 0; j < b; ++j)
                    for (index_type k = 0; k < c; ++k) CHECK(D(i, j, k) == T(i, j, k));
        }
        SECTION("permutation") {
            // P(k, i, j) = T(i, j, k)
            Dst D(c, a, b);
            auto P = T.template swizzle<TensorBase, 2, 0, 1>();
            D = P;
            for (index_type i = 0; i < a; ++i)
                for (index_type j = 0; j < b; ++j)
                    for (index_type k = 0; k < c; ++k) CHECK(D(k, i, j) == T(i, j, k));
        }
        SECTION("affine slice into a slice") {
            Dst D(a, b, c);
            for (index_type i = 0; i < a; ++i)
                for (index_type j = 0; j < b; ++j)
                    for (index_type k = 0; k < c; ++k) D(i, j, k) = -1.0;
            const auto &Tc = T;
            D.slice(zipper::slice(1, a - 2), full_extent_t{}, zipper::slice(0, c - 1)) =
                Tc.slice(zipper::slice(0, a - 2), full_extent_t{}, zipper::slice(1, c - 1));
            for (index_type i = 0; i < a; ++i)
                for (index_type j = 0; j < b; ++j)
                    for (index_type k = 0; k < c; ++k) {
                        const bool in = i >= 1 && i < a - 1 && k < c - 1;
                        CHECK(D(i, j, k) == (in ? T(i - 1, j, k + 1) : -1.0));
                    }
        }
        SECTION("rank-reducing slice") {
            // A rank-2 tensor view becomes a matrix explicitly.
            Matrix<double, dynamic_extent, dynamic_extent, !DstLM> M(a, c);
            M = as_matrix(T.slice(full_extent_t{}, index_type{b / 2}, full_extent_t{}));
            for (index_type i = 0; i < a; ++i)
                for (index_type k = 0; k < c; ++k) CHECK(M(i, k) == T(i, b / 2, k));
        }
    }
}

TEST_CASE("tiled relayout: large rank 2", "[assign][relayout]") {
    // 211 x 157 doubles (> 256 KiB per side): tiled, with ragged edge tiles.
    const auto M = filled<true>(211, 157);
    Matrix<double, dynamic_extent, dynamic_extent, false> C(211, 157);
    C = M;
    MatrixXX<double> T(157, 211);
    T = M.transpose();
    for (index_type i = 0; i < 211; ++i)
        for (index_type j = 0; j < 157; ++j) {
            CHECK(C(i, j) == M(i, j));
            CHECK(T(j, i) == M(i, j));
        }
}
