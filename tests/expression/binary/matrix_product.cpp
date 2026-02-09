
#include "../../catch_include.hpp"
#include "../../fmt_include.hpp"

#include <zipper/concepts/Matrix.hpp>
#include <zipper/expression/nullary/MDArray.hpp>
#include <zipper/expression/binary/MatrixProduct.hpp>

TEST_CASE("test_matrix_product", "[expression][binary][matrix_product]") {
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

    // MatrixProduct: a * a
    static_assert(zipper::concepts::MatrixExpression<decltype(a)>);
    binary::MatrixProduct mv(a, a);

    for (zipper::index_type j = 0; j < mv.extent(0); ++j) {
        for (zipper::index_type k = 0; k < mv.extent(1); ++k) {
            // Verify matrix product is correct: (a*a)(j,k) = sum_l a(j,l)*a(l,k)
            double expected = 0;
            for (zipper::index_type l = 0; l < a.extent(1); ++l) {
                expected += a(j, l) * a(l, k);
            }
            CHECK(mv(j, k) == expected);
        }
    }
}
