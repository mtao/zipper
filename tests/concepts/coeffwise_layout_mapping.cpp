#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/concepts/capabilities.hpp>

#include "../catch_include.hpp"

using namespace zipper;
namespace cc = zipper::expression::concepts;

namespace {
using DMat = Matrix<double, dynamic_extent, dynamic_extent>;

// Pull out the underlying expression type a wrapper holds.
template <typename Z>
using expr_t = std::remove_cvref_t<decltype(std::declval<Z>().expression())>;
}  // namespace

// A coefficient-wise op preserves the operand's index→linear layout/shape, so
// it advertises a forwarded layout mapping — but it is value-computing and owns
// no buffer, so it is NOT a LinearArray.  The kernel must not read its data().
TEST_CASE("coeff-wise layout traits: scalar 2*A keeps mapping, not LinearArray",
          "[concepts][layout]") {
    DMat A(3, 4);

    auto scaled = 2.0 * A;  // unary CoefficientWiseOperation
    using scaled_expr = expr_t<decltype(scaled)>;
    STATIC_CHECK(cc::HasLayoutMapping<scaled_expr>);
    STATIC_CHECK(!cc::LinearArray<scaled_expr>);

    // The forwarded mapping must describe the same shape/strides as A's.
    for (index_type i = 0; i < 3; ++i)
        for (index_type j = 0; j < 4; ++j)
            A(i, j) = static_cast<double>(i * 4 + j);

    const auto& op_e = scaled.expression();
    const auto& a_e = A.expression();
    auto op_map = op_e.mapping();
    auto a_map = a_e.mapping();

    CHECK(op_map.extents().extent(0) == a_map.extents().extent(0));
    CHECK(op_map.extents().extent(1) == a_map.extents().extent(1));
    for (rank_type r = 0; r < 2; ++r) CHECK(op_map.stride(r) == a_map.stride(r));
}

TEST_CASE("coeff-wise layout traits: A+B keeps mapping, not LinearArray",
          "[concepts][layout]") {
    DMat A(3, 4), B(3, 4);

    auto sum = A + B;  // binary Operation (dense, std::plus)
    using sum_expr = expr_t<decltype(sum)>;
    STATIC_CHECK(cc::HasLayoutMapping<sum_expr>);
    STATIC_CHECK(!cc::LinearArray<sum_expr>);

    for (index_type i = 0; i < 3; ++i)
        for (index_type j = 0; j < 4; ++j) {
            A(i, j) = static_cast<double>(i * 4 + j);
            B(i, j) = static_cast<double>(i + j);
        }

    const auto& op_e = sum.expression();
    const auto& a_e = A.expression();
    auto op_map = op_e.mapping();
    auto a_map = a_e.mapping();

    CHECK(op_map.extents().extent(0) == a_map.extents().extent(0));
    CHECK(op_map.extents().extent(1) == a_map.extents().extent(1));
    for (rank_type r = 0; r < 2; ++r) CHECK(op_map.stride(r) == a_map.stride(r));
}
