#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-overflow"
#include "catch_include.hpp"
#include "fmt_include.hpp"

#include <zipper/concepts/Matrix.hpp>
#include <zipper/expression/nullary/MDArray.hpp>
#include <zipper/expression/binary/ArithmeticExpressions.hpp>
#include <zipper/expression/binary/MatrixProduct.hpp>
#include <zipper/expression/unary/Cast.hpp>
#include <zipper/expression/unary/ScalarArithmetic.hpp>


TEST_CASE("test_storage", "[storage][dense]") {
    using namespace zipper::expression;

    // Create two 4x4 MDArrays
    nullary::MDArray<double, zipper::extents<4, 4>> a;
    using a_type = decltype(a);

    auto as = a.as_std_span();
    nullary::MDArray<double, zipper::extents<4, zipper::dynamic_extent>>
        b(zipper::extents<4, zipper::dynamic_extent>{4});

    fmt::print(
        "A size: {} {} {}\n", a.linear_accessor().container().size(),
        a_type::extents_traits::static_size,
        zipper::detail::ExtentsTraits<zipper::extents<4, 4>>::static_size);
    auto bs = b.as_std_span();

    {
        int j = 0;
        for (auto& v : bs) {
            v = j++;
        }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-overflow"
        fmt::print("{}\n", bs);
#pragma GCC diagnostic pop
    }

    for (zipper::index_type j = 0; j < a.extent(0); ++j) {
        for (zipper::index_type k = 0; k < a.extent(1); ++k) {
            double v = b(j, k);
            fmt::print("{} {} {}\n", j, k, v);
            a(j, k) = b(j, k);
        }
    }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-overflow"
    fmt::print("{}\n", as);
#pragma GCC diagnostic pop

    // ScalarMultiplies: 2.0 * a
    unary::ScalarMultiplies<double, decltype(a), false> spv(2.0, a);

    for (zipper::index_type j = 0; j < spv.extent(0); ++j) {
        for (zipper::index_type k = 0; k < spv.extent(1); ++k) {
            fmt::print("{} ", spv(j, k));
        }
        fmt::print("\n");
    }

    // Binary Plus: a + b
    binary::Plus<decltype(a), decltype(b)> av(a, b);

    for (zipper::index_type j = 0; j < av.extent(0); ++j) {
        for (zipper::index_type k = 0; k < av.extent(1); ++k) {
            fmt::print("{} {} {}\n", j, k, av(j, k));
        }
    }

    // Binary Multiplies (element-wise): a * a
    binary::Multiplies<decltype(a), decltype(a)> pv(a, a);

    for (zipper::index_type j = 0; j < pv.extent(0); ++j) {
        for (zipper::index_type k = 0; k < pv.extent(1); ++k) {
            fmt::print("{} {} {}\n", j, k, pv(j, k));
        }
    }

    // MatrixProduct: a * a
    static_assert(zipper::concepts::MatrixExpression<decltype(a)>);
    binary::MatrixProduct mv(a, a);

    for (zipper::index_type j = 0; j < mv.extent(0); ++j) {
        for (zipper::index_type k = 0; k < mv.extent(1); ++k) {
            fmt::print("{} ", mv(j, k));
        }
        fmt::print("\n");
    }

    // Cast: double -> int
    auto cv = unary::cast<int>(a);

    for (zipper::index_type j = 0; j < cv.extent(0); ++j) {
        for (zipper::index_type k = 0; k < cv.extent(1); ++k) {
            fmt::print("{} ", cv(j, k));
        }
        fmt::print("\n");
    }

    // Assign from expression
    b.assign(pv);
    fmt::print("B extent: {}\n", b.extent(1));
    for (zipper::index_type j = 0; j < b.extent(0); ++j) {
        for (zipper::index_type k = 0; k < b.extent(1); ++k) {
            fmt::print("{} ", b(j, k));
        }
        fmt::print("\n");
    }
}

#pragma GCC diagnostic pop
