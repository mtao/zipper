#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-overflow"
#include "../catch_include.hpp"

#include <zipper/concepts/Matrix.hpp>
#include <zipper/expression/nullary/MDArray.hpp>
#include <zipper/expression/binary/ArithmeticExpressions.hpp>
#include <zipper/expression/binary/MatrixProduct.hpp>
#include <zipper/expression/unary/Cast.hpp>
#include <zipper/expression/unary/ScalarArithmetic.hpp>

#include <print>

TEST_CASE("test_storage", "[storage][dense]") {
    using namespace zipper::expression;

    // Create two 4x4 MDArrays
    nullary::MDArray<double, zipper::extents<4, 4>> a;
    using a_type = decltype(a);

    auto as = a.as_std_span();
    nullary::MDArray<double, zipper::extents<4, zipper::dynamic_extent>>
        b(zipper::extents<4, zipper::dynamic_extent>{4});

    std::println(
        "A size: {} {} {}", a.linear_accessor().container().size(),
        a_type::extents_traits::static_size,
        zipper::detail::ExtentsTraits<zipper::extents<4, 4>>::static_size);
    auto bs = b.as_std_span();

    {
        int j = 0;
        for (auto& v : bs) {
            v = j++;
        }
        std::println("{}", bs);
    }

    for (zipper::index_type j = 0; j < a.extent(0); ++j) {
        for (zipper::index_type k = 0; k < a.extent(1); ++k) {
            double v = b(j, k);
            std::println("{} {} {}", j, k, v);
            a(j, k) = b(j, k);
        }
    }
    std::println("{}", as);

    // ScalarMultiplies: 2.0 * a
    unary::ScalarMultiplies<double, decltype(a), false> spv(2.0, a);

    for (zipper::index_type j = 0; j < spv.extent(0); ++j) {
        for (zipper::index_type k = 0; k < spv.extent(1); ++k) {
            std::print("{} ", spv(j, k));
        }
        std::println("");
    }

    // Binary Plus: a + b
    binary::Plus<decltype(a), decltype(b)> av(a, b);

    for (zipper::index_type j = 0; j < av.extent(0); ++j) {
        for (zipper::index_type k = 0; k < av.extent(1); ++k) {
            std::println("{} {} {}", j, k, av(j, k));
        }
    }

    // Binary Multiplies (element-wise): a * a
    binary::Multiplies<decltype(a), decltype(a)> pv(a, a);

    for (zipper::index_type j = 0; j < pv.extent(0); ++j) {
        for (zipper::index_type k = 0; k < pv.extent(1); ++k) {
            std::println("{} {} {}", j, k, pv(j, k));
        }
    }

    // MatrixProduct: a * a
    STATIC_CHECK(zipper::concepts::MatrixExpression<decltype(a)>);
    binary::MatrixProduct mv(a, a);

    for (zipper::index_type j = 0; j < mv.extent(0); ++j) {
        for (zipper::index_type k = 0; k < mv.extent(1); ++k) {
            std::print("{} ", mv(j, k));
        }
        std::println("");
    }

    // Cast: double -> int
    auto cv = unary::cast<int>(a);

    for (zipper::index_type j = 0; j < cv.extent(0); ++j) {
        for (zipper::index_type k = 0; k < cv.extent(1); ++k) {
            std::print("{} ", cv(j, k));
        }
        std::println("");
    }

    // Assign from expression
    b.assign(pv);
    std::println("B extent: {}", b.extent(1));
    for (zipper::index_type j = 0; j < b.extent(0); ++j) {
        for (zipper::index_type k = 0; k < b.extent(1); ++k) {
            std::print("{} ", b(j, k));
        }
        std::println("");
    }
}

#pragma GCC diagnostic pop
