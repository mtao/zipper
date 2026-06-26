#include <type_traits>

#include <zipper/Matrix.hpp>
#include <zipper/expression/concepts/capabilities.hpp>

#include "../catch_include.hpp"

using namespace zipper;
namespace cc = zipper::expression::concepts;

namespace {
using DM = Matrix<double, dynamic_extent, dynamic_extent>;
}

// PROTOTYPE: a coeff-wise op exposes a flat, mapping-free operator[] that
// composes the op over the child's buffer order. This is what lets a linear
// assignment `to[k] = (2*A)[k]` auto-vectorize (it bottoms out at
// 2 * A.data()[k]). The op is still NOT a LinearArray (no buffer of its own).
TEST_CASE("coeffwise: scalar op is flat-indexable (not a LinearArray)",
          "[coeffwise][vectorize]") {
    DM A(4, 5);
    for (index_type i = 0; i < 4; ++i)
        for (index_type j = 0; j < 5; ++j) A(i, j) = static_cast<double>(i * 5 + j);

    auto e = (2.0 * A).expression();  // ScalarOperation over A
    using E = std::remove_cvref_t<decltype(e)>;

    // No buffer of its own, but now flat-indexable by value.
    STATIC_CHECK(!cc::LinearArray<E>);
    STATIC_CHECK(requires(const E& ce, index_type k) { ce[k]; });

    // Flat operator[] composes the op over the child's (row-major) buffer.
    for (index_type k = 0; k < 20; ++k)
        CHECK(e[k] == 2.0 * A(k / 5, k % 5));
}

// Deeper coeff-wise trees compose: the flat operator[] propagates through
// nested scalar/binary ops, so `4*A+B` and `A+B+C+D` are flat-vectorizable too
// (every leaf is contiguous). This is the FlatVectorizable recursion: a node is
// flat iff its operands are.
TEST_CASE("coeffwise: deeper trees are flat-vectorizable (4*A+B, A+B+C+D)",
          "[coeffwise][vectorize]") {
    DM A(4, 5), B(4, 5), C(4, 5), D(4, 5);
    for (index_type i = 0; i < 4; ++i)
        for (index_type j = 0; j < 5; ++j) {
            double v = static_cast<double>(i * 5 + j);
            A(i, j) = v;
            B(i, j) = 2.0 * v + 1.0;
            C(i, j) = -v;
            D(i, j) = 0.5 * v;
        }

    // 4*A + B : binary op whose lhs is a scalar op — both operands flat.
    {
        auto e = (4.0 * A + B).expression();
        using E = std::remove_cvref_t<decltype(e)>;
        STATIC_CHECK(cc::FlatVectorizable<E>);
        STATIC_CHECK(!cc::LinearArray<E>);  // still owns no buffer
        for (index_type k = 0; k < 20; ++k) {
            double a = A(k / 5, k % 5), b = B(k / 5, k % 5);
            CHECK(e[k] == 4.0 * a + b);
        }
    }

    // A + B + C + D : left-associated chain of binary ops, depth 3.
    {
        auto e = (A + B + C + D).expression();
        using E = std::remove_cvref_t<decltype(e)>;
        STATIC_CHECK(cc::FlatVectorizable<E>);
        for (index_type k = 0; k < 20; ++k) {
            index_type i = k / 5, j = k % 5;
            CHECK(e[k] == A(i, j) + B(i, j) + C(i, j) + D(i, j));
        }
    }

    // 4*(A+B) : scalar op over a binary op — nesting in the other direction.
    {
        auto e = (4.0 * (A + B)).expression();
        using E = std::remove_cvref_t<decltype(e)>;
        STATIC_CHECK(cc::FlatVectorizable<E>);
        for (index_type k = 0; k < 20; ++k) {
            index_type i = k / 5, j = k % 5;
            CHECK(e[k] == 4.0 * (A(i, j) + B(i, j)));
        }
    }
}
