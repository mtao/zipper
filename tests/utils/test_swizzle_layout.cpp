#include <type_traits>

#include <zipper/Matrix.hpp>
#include <zipper/expression/concepts/capabilities.hpp>

#include "../catch_include.hpp"

using namespace zipper;
namespace cc = zipper::expression::concepts;

namespace {
using DMat = Matrix<double, dynamic_extent, dynamic_extent>;

template <typename Z>
using expr_t = std::remove_cvref_t<decltype(std::declval<Z>().expression())>;
}  // namespace

// A swizzle (transpose) of dense storage forwards the layout-mapping and
// linear-array capabilities from its child — verified without any GEMM kernel.
TEST_CASE("swizzle: transpose is a LinearArray", "[swizzle][layout]") {
    DMat M(4, 6);
    for (index_type i = 0; i < 4; ++i)
        for (index_type j = 0; j < 6; ++j)
            M(i, j) = static_cast<double>(i * 6 + j);

    auto t = M.transpose();
    using TExpr = expr_t<decltype(t)>;
    STATIC_CHECK(cc::HasLayoutMapping<TExpr>);
    STATIC_CHECK(cc::LinearArray<TExpr>);

    const auto& s = t.expression();

    // Extents swapped.
    REQUIRE(t.extent(0) == M.extent(1));
    REQUIRE(t.extent(1) == M.extent(0));

    // The LinearArray contract: s[ s.mapping()(i,j) ] == s(i,j) == M(j,i).
    auto map = s.mapping();
    for (index_type i = 0; i < t.extent(0); ++i)
        for (index_type j = 0; j < t.extent(1); ++j) {
            CHECK(s[map(i, j)] == M(j, i));
            CHECK(t(i, j) == M(j, i));
        }

    // Transpose of row-major is column-major: unit stride down dim 0.
    CHECK(map.stride(0) == 1);
    CHECK(map.stride(1) == M.extent(1));
}
