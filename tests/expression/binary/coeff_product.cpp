
#include "../../catch_include.hpp"
#include "../../fmt_include.hpp"

#include <zipper/expression/nullary/MDArray.hpp>
#include <zipper/expression/binary/ArithmeticExpressions.hpp>

TEST_CASE("test_coeff_product", "[expression][binary][coeff_product]") {
    using namespace zipper::expression;

    // Create a 4x4 MDArray and fill it
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

    // Binary Multiplies (element-wise): a * a
    binary::Multiplies<decltype(a), decltype(a)> pv(a, a);

    for (zipper::index_type j = 0; j < pv.extent(0); ++j) {
        for (zipper::index_type k = 0; k < pv.extent(1); ++k) {
            CHECK(pv(j, k) == a(j, k) * a(j, k));
        }
    }
}
