#include <type_traits>

#include <zipper/Matrix.hpp>
#include <zipper/Tensor.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/concepts/capabilities.hpp>
#include <zipper/storage/layout_traits.hpp>

#include "../catch_include.hpp"

using namespace zipper;
namespace cc = zipper::expression::concepts;

namespace {
using DMat = Matrix<double, dynamic_extent, dynamic_extent>;

template <typename Z>
using expr_t = std::remove_cvref_t<decltype(std::declval<Z>().expression())>;

DMat filled(index_type r, index_type c) {
    DMat M(r, c);
    for (index_type i = 0; i < r; ++i)
        for (index_type j = 0; j < c; ++j)
            M(i, j) = static_cast<double>(i * 100 + j);
    return M;
}

// Verify the LinearArray contract on a rank-1 slice view: it satisfies the
// concept and view[ view.mapping()(k) ] == view(k).
template <typename View>
void check_linear_contract(const View& v) {
    STATIC_CHECK(cc::HasLayoutMapping<std::remove_cvref_t<View>>);
    STATIC_CHECK(cc::LinearArray<std::remove_cvref_t<View>>);
    auto map = v.mapping();
    for (index_type k = 0; k < v.extent(0); ++k) CHECK(v[map(k)] == v(k));
}
}  // namespace

TEST_CASE("slice: row/col of dense storage are LinearArrays", "[slice][layout]") {
    DMat M = filled(5, 7);

    SECTION("row slice (contiguous, stride 1)") {
        auto r = M.row(index_type{2});  // VectorBase<Slice>
        const auto& s = r.expression();
        check_linear_contract(s);
        REQUIRE(s.extent(0) == M.extent(1));
        for (index_type j = 0; j < 7; ++j) CHECK(s(j) == M(2, j));
        CHECK(s.mapping().stride(0) == 1);  // contiguous within the row
    }

    SECTION("column slice (strided by ncols)") {
        auto c = M.col(index_type{3});
        const auto& s = c.expression();
        check_linear_contract(s);
        REQUIRE(s.extent(0) == M.extent(0));
        for (index_type i = 0; i < 5; ++i) CHECK(s(i) == M(i, 3));
        CHECK(s.mapping().stride(0) == M.extent(1));  // down a column
    }
}

// Gathers (index lists) are not affine: no single stride describes them, so
// they must not claim a layout mapping (regression: they once did, and a
// buffer-to-buffer copy through them read the wrong elements).
TEST_CASE("slice layout: gathers are not linear arrays", "[slice][layout]") {
    DMat M = filled(5, 6);
    auto gv = M.slice(full_extent_t{}, std::vector<index_type>{4, 0, 2});
    auto ga = M.slice(std::array<index_type, 2>{3, 1}, full_extent_t{});
    STATIC_CHECK_FALSE(cc::HasLayoutMapping<expr_t<decltype(gv)>>);
    STATIC_CHECK_FALSE(cc::LinearArray<expr_t<decltype(gv)>>);
    STATIC_CHECK_FALSE(cc::LinearArray<expr_t<decltype(ga)>>);
    STATIC_CHECK_FALSE(cc::LinearArray<expr_t<decltype(gv.transpose())>>);
    // Affine slices still are, including through a transpose.
    auto blk = M.slice(zipper::slice(1, 3), zipper::slice(2, 2));
    STATIC_CHECK(cc::LinearArray<expr_t<decltype(blk)>>);
    STATIC_CHECK(cc::LinearArray<expr_t<decltype(blk.transpose())>>);
}

// A transpose (Swizzle<E,1,0>) of dense storage shares the buffer with swapped
// strides, so it is still a LinearArray and honours the operator[] contract.
TEST_CASE("layout traits: transpose of dense storage is a LinearArray",
          "[concepts][layout]") {
    DMat M(3, 4);
    for (index_type i = 0; i < 3; ++i)
        for (index_type j = 0; j < 4; ++j) M(i, j) = static_cast<double>(i * 4 + j);
    auto T = M.transpose();
    STATIC_CHECK(cc::LinearArray<expr_t<decltype(T)>>);
    const auto& e = T.expression();
    auto map = e.mapping();
    CHECK(map.stride(0) == M.expression().mapping().stride(1));
    CHECK(map.stride(1) == M.expression().mapping().stride(0));
    for (index_type i = 0; i < 4; ++i)
        for (index_type j = 0; j < 3; ++j) {
            CHECK(e[map(i, j)] == e(i, j));
            CHECK(e(i, j) == M(j, i));
        }

    // A lazy child stays non-linear through the transpose.
    auto lazy = (M + M).transpose();
    STATIC_CHECK(!cc::LinearArray<expr_t<decltype(lazy)>>);
}

// Views keep their child's layout in their mapping type: a block of
// row-major storage is (padded) row-major, a transpose is a permuted layout,
// and only irregular slices fall back to layout_stride.
TEST_CASE("view layout: mapping types and fastest dimension",
          "[slice][swizzle][layout]") {
    namespace zs = zipper::storage;
    auto fastest = [](const auto &z) {
        return zs::fastest_dimension_v<
            std::remove_cvref_t<decltype(z.expression().mapping())>>;
    };
    using DMatCol = Matrix<double, dynamic_extent, dynamic_extent, false>;
    DMat M = filled(4, 5);
    DMatCol C(4, 5);
    Tensor<double, dynamic_extent, dynamic_extent, dynamic_extent> T(2, 3, 4);

    // Transposes swap it; a transpose of a transpose restores it.
    CHECK(fastest(M.transpose()) == 0);
    CHECK(fastest(C.transpose()) == 1);
    CHECK(fastest(M.transpose().transpose()) == 1);
    CHECK(fastest(T.swizzle<TensorBase, 2, 0, 1>()) == 0);

    // Slices keep it, renumbered past indexed-away dimensions.
    CHECK(fastest(M.slice(zipper::slice(1, 2), zipper::slice(0, 3))) == 1);
    CHECK(fastest(M.row(1)) == 0);
    CHECK(fastest(T.slice(index_type{1}, full_extent_t{}, full_extent_t{})) == 1);
    // ... unless the fastest dimension itself is indexed away (layout_stride).
    CHECK(fastest(M.col(1)) == zs::unknown_dimension);

    // The standard submdspan layouts: a row block of row-major storage is
    // still layout_right; a column block is layout_right_padded.
    using rows_t = std::remove_cvref_t<decltype(
        M.slice(zipper::slice(1, 2), full_extent_t{}).expression().mapping())>;
    using cols_t = std::remove_cvref_t<decltype(
        M.slice(full_extent_t{}, zipper::slice(1, 2)).expression().mapping())>;
    STATIC_CHECK(std::is_same_v<typename rows_t::layout_type, zs::layout_right>);
    STATIC_CHECK_FALSE(std::is_same_v<typename cols_t::layout_type, zs::layout_stride>);
    STATIC_CHECK(zs::fastest_dimension_v<cols_t> == 1);
}
