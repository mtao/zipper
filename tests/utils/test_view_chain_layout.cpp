// Chained reordering views compose into a SINGLE mapping over the root
// buffer: swizzle∘slice / slice∘swizzle / reshape chains each forward
// data()/operator[] to the root storage and compose their stride transform
// into one layout mapping, so a consumer pays one index transform per
// element — not a nested coeff walk per level.

#include <type_traits>

#include <zipper/Matrix.hpp>
#include <zipper/expression/concepts/capabilities.hpp>
#include <zipper/expression/unary/Reshape.hpp>

#include "../catch_include.hpp"

using namespace zipper;
namespace cc = zipper::expression::concepts;

namespace {
using DMat = Matrix<double, dynamic_extent, dynamic_extent>;

DMat filled(index_type r, index_type c) {
    DMat M(r, c);
    for (index_type i = 0; i < r; ++i)
        for (index_type j = 0; j < c; ++j)
            M(i, j) = static_cast<double>(i * 100 + j);
    return M;
}
}  // namespace

TEST_CASE("view chains compose to one mapping over the root buffer",
          "[layout][composition]") {
    DMat M = filled(6, 8);

    SECTION("transpose of a sub-block (slice then swizzle)") {
        auto sub = M.slice(zipper::slice(1, 4), zipper::slice(2, 5));
        auto t = sub.transpose();
        const auto& s = t.expression();
        using S = std::remove_cvref_t<decltype(s)>;
        STATIC_CHECK(cc::HasLayoutMapping<S>);
        STATIC_CHECK(cc::LinearArray<S>);

        // Same root buffer (slice folds its origin into the pointer);
        // the two stride transforms composed into one map.
        CHECK(s.data() == M.expression().data() + (1 * 8 + 2));
        auto map = s.mapping();
        for (index_type i = 0; i < t.extent(0); ++i)
            for (index_type j = 0; j < t.extent(1); ++j) {
                CHECK(s[map(i, j)] == M(1 + j, 2 + i));
                CHECK(s(i, j) == M(1 + j, 2 + i));
            }
    }

    SECTION("sub-block of a transpose (swizzle then slice)") {
        auto t = M.transpose();
        auto sub = t.slice(zipper::slice(1, 3), zipper::slice(2, 4));
        const auto& s = sub.expression();
        using S = std::remove_cvref_t<decltype(s)>;
        STATIC_CHECK(cc::LinearArray<S>);
        // Origin (1,2) in the transposed view = (2,1) in M = offset 2*8+1.
        CHECK(s.data() == M.expression().data() + (2 * 8 + 1));
        auto map = s.mapping();
        for (index_type i = 0; i < sub.extent(0); ++i)
            for (index_type j = 0; j < sub.extent(1); ++j)
                CHECK(s[map(i, j)] == M(2 + j, 1 + i));
    }

    SECTION("reshape of a full row-major matrix composes with flat consumers") {
        Matrix<double, 3, 4> A{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}};
        auto r = expression::unary::Reshape(A.expression(),
                                            zipper::extents<4, 3>{});
        using R = std::remove_cvref_t<decltype(r)>;
        STATIC_CHECK(cc::LinearArray<R>);
        STATIC_CHECK(cc::FlatVectorizable<R>);
        CHECK(r.data() == A.expression().data());
        auto map = r.mapping();
        for (index_type i = 0; i < 4; ++i)
            for (index_type j = 0; j < 3; ++j)
                CHECK(r[map(i, j)] == r(i, j));
    }
}
