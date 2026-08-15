#include <type_traits>

#include <zipper/Matrix.hpp>
#include <zipper/expression/concepts/capabilities.hpp>

#include "../catch_include.hpp"

using namespace zipper;
namespace cc = zipper::expression::concepts;

namespace {
using DM = Matrix<double, dynamic_extent, dynamic_extent>;             // row-major
using DMc = Matrix<double, dynamic_extent, dynamic_extent, false>;     // col-major

template <typename E>
using flat_layout_of =
    typename zipper::expression::detail::ExpressionTraits<E>::flat_layout_type;
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

// The flat path is layout-GENERIC: a column-major tree is flat-vectorizable in
// its own (layout_left) order, and a column-major target is flat-COMPATIBLE
// with a column-major source — exactly the "everything has a consistent
// mapping" condition. Mixed layouts (row-major target ← column-major source)
// are NOT flat-compatible and fall back to the index walk (still correct).
TEST_CASE("coeffwise: flat-vectorization supports >1 mapping type",
          "[coeffwise][vectorize][layout]") {
    namespace st = zipper::storage;
    using RowExpr = std::remove_cvref_t<decltype((4.0 * std::declval<DM&>() +
                                                  std::declval<DM&>())
                                                     .expression())>;
    using ColExpr = std::remove_cvref_t<decltype((4.0 * std::declval<DMc&>() +
                                                  std::declval<DMc&>())
                                                     .expression())>;
    using RowTarget = std::remove_cvref_t<decltype(std::declval<DM&>().expression())>;
    using ColTarget = std::remove_cvref_t<decltype(std::declval<DMc&>().expression())>;

    // Each tree carries the layout it actually enumerates in — not hardcoded.
    STATIC_CHECK(std::is_same_v<flat_layout_of<RowExpr>, st::layout_right>);
    STATIC_CHECK(std::is_same_v<flat_layout_of<ColExpr>, st::layout_left>);

    // Same layout on both sides ⇒ flat-compatible (vectorized fast path).
    STATIC_CHECK(cc::FlatCompatible<RowTarget, RowExpr>);
    STATIC_CHECK(cc::FlatCompatible<ColTarget, ColExpr>);
    // Mismatched layouts ⇒ NOT flat-compatible ⇒ safe fallback.
    STATIC_CHECK(!cc::FlatCompatible<RowTarget, ColExpr>);
    STATIC_CHECK(!cc::FlatCompatible<ColTarget, RowExpr>);
}

// End-to-end: the actual assignment (through AssignHelper) produces correct
// results on BOTH the vectorized fast path (matching layout) and the fallback
// (mixed layout). Same numeric answer either way.
TEST_CASE("coeffwise: linear assignment fast path is correct (any layout)",
          "[coeffwise][vectorize][assign]") {
    auto fill = [](auto& M) {
        for (index_type i = 0; i < 4; ++i)
            for (index_type j = 0; j < 5; ++j)
                M(i, j) = static_cast<double>(i * 5 + j) - 7.0;
    };

    DM Ar(4, 5), Br(4, 5);
    DMc Ac(4, 5), Bc(4, 5);
    fill(Ar); fill(Br); fill(Ac); fill(Bc);

    // Row target ← row source: fast path.
    DM Cr(4, 5);
    Cr = 4.0 * Ar + Br;
    // Col target ← col source: fast path (column-major).
    DMc Cc(4, 5);
    Cc = 4.0 * Ac + Bc;
    // Row target ← col source: fallback (layout mismatch), must still be right.
    DM Cmix(4, 5);
    Cmix = 4.0 * Ac + Bc;

    for (index_type i = 0; i < 4; ++i)
        for (index_type j = 0; j < 5; ++j) {
            double want = 4.0 * (static_cast<double>(i * 5 + j) - 7.0) +
                          (static_cast<double>(i * 5 + j) - 7.0);
            CHECK(Cr(i, j) == want);
            CHECK(Cc(i, j) == want);
            CHECK(Cmix(i, j) == want);
        }
}
