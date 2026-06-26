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
