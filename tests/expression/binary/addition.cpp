
#include "../../catch_include.hpp"
#include "../../fmt_include.hpp"

#include <zipper/expression/nullary/MDArray.hpp>
#include <zipper/expression/binary/ArithmeticExpressions.hpp>

TEST_CASE("test_addition", "[expression][binary][addition]") {
    using namespace zipper::expression;

    // Create two 4x4 MDArrays
    nullary::MDArray<double, zipper::extents<4, 4>> a;
    nullary::MDArray<double, zipper::extents<4, zipper::dynamic_extent>>
        b(zipper::extents<4, zipper::dynamic_extent>{4});

    auto bs = b.as_std_span();
    {
        int j = 0;
        for (auto& v : bs) {
            v = j++;
        }
    }

    for (zipper::index_type j = 0; j < a.extent(0); ++j) {
        for (zipper::index_type k = 0; k < a.extent(1); ++k) {
            a(j, k) = b(j, k);
        }
    }

    // Binary Plus: a + b
    binary::Plus<decltype(a), decltype(b)> av(a, b);

    for (zipper::index_type j = 0; j < av.extent(0); ++j) {
        for (zipper::index_type k = 0; k < av.extent(1); ++k) {
            CHECK(av(j, k) == a(j, k) + b(j, k));
        }
    }
}
