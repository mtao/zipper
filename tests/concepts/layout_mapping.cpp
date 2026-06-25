#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/concepts/capabilities.hpp>

#include "../catch_include.hpp"

using namespace zipper;
namespace cc = zipper::expression::concepts;

namespace {
using DMat = Matrix<double, dynamic_extent, dynamic_extent>;

// Pull out the underlying expression type a wrapper holds.
template <typename Z>
using expr_t = std::remove_cvref_t<decltype(std::declval<Z>().expression())>;
}  // namespace

// Dense storage is both a layout-mapped expression and a linear array.
TEST_CASE("layout traits: dense storage is a LinearArray", "[concepts][layout]") {
    STATIC_CHECK(cc::HasLayoutMapping<expr_t<DMat>>);
    STATIC_CHECK(cc::LinearArray<expr_t<DMat>>);
    STATIC_CHECK(cc::HasLayoutMapping<expr_t<Matrix<double, 3, 4>>>);
    STATIC_CHECK(cc::LinearArray<expr_t<Vector<double, 5>>>);

    // The operator[] / mapping() contract: e[ mapping()(i,j) ] == e(i,j).
    DMat M(3, 4);
    for (index_type i = 0; i < 3; ++i)
        for (index_type j = 0; j < 4; ++j) M(i, j) = static_cast<double>(i * 4 + j);
    const auto& e = M.expression();
    auto map = e.mapping();
    for (index_type i = 0; i < 3; ++i)
        for (index_type j = 0; j < 4; ++j) CHECK(e[map(i, j)] == e(i, j));
}

// A lazy, value-computing expression is NOT a linear array (no buffer), even
// once it gains a forwarded mapping later — the kernel must not read its data.
TEST_CASE("layout traits: lazy expressions are not LinearArray",
          "[concepts][layout]") {
    DMat A(2, 2), B(2, 2);
    auto sum = A + B;  // lazy binary expression
    STATIC_CHECK(!cc::LinearArray<expr_t<decltype(sum)>>);
}
