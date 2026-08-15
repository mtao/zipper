#include <type_traits>

#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/concepts/capabilities.hpp>

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
