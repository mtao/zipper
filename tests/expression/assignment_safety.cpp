#include <complex>
// Intentionally first: this header must also work from inside AssignHelper,
// before MDArray/MDSpan or any constant template is defined.
#include <zipper/expression/detail/AssignmentSafety.hpp>

#include "catch_include.hpp"
#include <zipper/Matrix.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/MDArray.hpp>
#include <zipper/expression/nullary/MDSpan.hpp>
#include <zipper/expression/nullary/StaticConstant.hpp>
#include <zipper/expression/nullary/Zero.hpp>

namespace {
using zipper::expression::detail::assignment_is_safe;
using zipper::expression::nullary::MDArray;
using zipper::expression::nullary::MDSpan;
using zipper::expression::nullary::Constant;
using zipper::expression::nullary::StaticConstant;
using zipper::expression::nullary::Zero;
using zipper::expression::unary::CoefficientWiseOperation;
using zipper::expression::unary::ScalarOperation;
using zipper::expression::binary::Operation;
using zipper::extents;

struct PureNegate {
    auto operator()(double value) const -> double { return -value; }
};

struct OpaqueValue {
    double value;
    auto operator-() const -> OpaqueValue { return {-value}; }
};

struct PretendLeaf {
    static constexpr bool is_coefficient_consistent = true;
    auto data() const -> const double * { return nullptr; }
};
} // namespace

template <>
struct zipper::expression::detail::is_pointwise_operation<PureNegate, double>
    : std::true_type {};

TEST_CASE("assignment_safety_dense_pointwise", "[expression][assignment_safety]") {
    MDArray<double, extents<2, 3>> a, b;
    const auto &read_a = a;
    CHECK(assignment_is_safe(a, a));
    CHECK(assignment_is_safe(read_a, a));
    CHECK(assignment_is_safe(b, a));

    const CoefficientWiseOperation neg(a, std::negate<double>{});
    CHECK(assignment_is_safe(neg, a));
    const Operation sum(a, b, std::plus<double>{});
    CHECK(assignment_is_safe(sum, a));
    const ScalarOperation scale(sum, 2.0, std::multiplies<double>{});
    CHECK(assignment_is_safe(scale, a));
    const ScalarOperation subtract(3.0, a, std::minus<>{});
    CHECK(assignment_is_safe(subtract, a));
    using RefScalar = ScalarOperation<decltype(a) &, std::plus<>, double &, true>;
    RefScalar reference_capture(a, a.coeff_ref(0, 0));
    CHECK_FALSE(assignment_is_safe(reference_capture, a));
    const CoefficientWiseOperation opted_in(a, PureNegate{});
    CHECK(assignment_is_safe(opted_in, a));

    MDArray<double, extents<2, 3>, zipper::storage::layout_left> independent_left;
    CHECK(assignment_is_safe(independent_left, a));
    CHECK(assignment_is_safe(a, independent_left));
}

// Views may share storage with anything; the proof never looks at addresses,
// so no view is proven, however its memory is placed. `noalias()` is the
// caller's statement that a view-based assignment is independent.
TEST_CASE("assignment_safety_views_are_not_proven", "[expression][assignment_safety]") {
    std::array<double, 10> data{};
    using Span = MDSpan<double, extents<4>>;
    Span a(std::span<double, 4>(data.data(), 4));
    Span same(std::span<double, 4>(data.data(), 4));
    Span shifted(std::span<double, 4>(data.data() + 1, 4));
    Span adjacent(std::span<double, 4>(data.data() + 4, 4));
    MDSpan<const double, extents<4>> read(std::span<const double, 4>(data.data(), 4));
    CHECK_FALSE(assignment_is_safe(same, a));
    CHECK_FALSE(assignment_is_safe(read, a));
    CHECK_FALSE(assignment_is_safe(a, read));  // const destination
    CHECK_FALSE(assignment_is_safe(shifted, a));
    CHECK_FALSE(assignment_is_safe(adjacent, a));
    const Operation sum(read, adjacent, std::plus<>{});
    CHECK_FALSE(assignment_is_safe(sum, a));

    // An owning destination does not make a view source safe (the view may
    // look into the destination), nor the other way round.
    MDArray<double, extents<4>> owned;
    CHECK_FALSE(assignment_is_safe(adjacent, owned));
    CHECK_FALSE(assignment_is_safe(owned, a));

    // Generators read nothing, so they are safe into views.
    CHECK(assignment_is_safe(Zero<double, 4>{}, a));
    CHECK(assignment_is_safe(Constant<double>(1.0), a));
    const ScalarOperation scaled(Constant<double, 4>(1.0), 2.0,
                                 std::multiplies<double>{});
    CHECK(assignment_is_safe(scaled, a));

    // Slices and transposes of owning arrays are views too.
    zipper::Matrix<double, zipper::dynamic_extent, zipper::dynamic_extent> m(4, 5);
    auto blk = m.slice(zipper::slice(1, 2), zipper::slice(0, 3));
    CHECK(assignment_is_safe(Zero<double, 2, 3>{}, blk.expression()));
    CHECK_FALSE(assignment_is_safe(blk.expression(), m.expression()));
}

TEST_CASE("assignment_safety_is_a_compile_time_property",
          "[expression][assignment_safety]") {
    using zipper::expression::detail::assignment_is_safe_v;
    using A = MDArray<double, zipper::dextents<2>>;
    using S = MDSpan<double, zipper::dextents<2>>;
    STATIC_CHECK(assignment_is_safe_v<A, A>);
    STATIC_CHECK(assignment_is_safe_v<const A &, A>);
    STATIC_CHECK(assignment_is_safe_v<Operation<const A &, const A &, std::plus<double>>, A>);
    STATIC_CHECK_FALSE(assignment_is_safe_v<S, A>);
    STATIC_CHECK_FALSE(assignment_is_safe_v<A, S>);
    STATIC_CHECK(assignment_is_safe_v<Zero<double>, S>);
}

TEST_CASE("assignment_safety_shape_and_constants", "[expression][assignment_safety]") {
    MDArray<double, zipper::dextents<2>> a(zipper::dextents<2>(2, 3));
    MDArray<double, zipper::dextents<2>> b(zipper::dextents<2>(3, 2));
    // Distinct owning arrays never overlap; a shape change is a resize of
    // the destination before evaluation, which the proof covers.
    CHECK(assignment_is_safe(b, a));
    CHECK(assignment_is_safe(Constant<double>(4.0), a));
    CHECK(assignment_is_safe(Constant<double, 2, 3>(4.0), a));
    CHECK(assignment_is_safe(Constant<double, 3, 2>(4.0), a));
    CHECK(assignment_is_safe(StaticConstant<double, 0.0>{}, a));
    CHECK(assignment_is_safe(StaticConstant<double, 1.0, 2, 3>{}, a));
    CHECK(assignment_is_safe(StaticConstant<double, 1.0, 3, 2>{}, a));
    // Zero reads nothing: as a source it is safe whatever its element type.
    CHECK(assignment_is_safe(Zero<double, 2, 3>{}, a));
    CHECK(assignment_is_safe(Zero<double>{}, a));
    CHECK(assignment_is_safe(Zero<double, 3, 2>{}, a));
    // Destinations are only recognized for arithmetic element types (DenseLeaf),
    // so assignments into e.g. std::complex storage are never proven here.
    MDArray<std::complex<double>, zipper::dextents<2>> c(zipper::dextents<2>(2, 3));
    CHECK_FALSE(assignment_is_safe(Zero<std::complex<double>, 2, 3>{}, c));
    const Constant<double> scalar(2.0);
    const Operation sum(a, scalar, std::plus<>{});
    CHECK(assignment_is_safe(sum, a));

    MDArray<double, extents<>> rank_zero;
    CHECK(assignment_is_safe(rank_zero, rank_zero));
    CHECK(assignment_is_safe(rank_zero, a));  // broadcast from another array
    MDArray<double, zipper::dextents<1>> empty(zipper::dextents<1>(0));
    CHECK(assignment_is_safe(empty, empty));
    MDSpan<double, zipper::dextents<1>> empty_span(std::span<double>{});
    CHECK_FALSE(assignment_is_safe(empty_span, empty));  // views: never
}

TEST_CASE("assignment_safety_opaque_dependencies", "[expression][assignment_safety]") {
    MDArray<double, extents<4>> a, b;
    int calls = 0;
    const auto callback = [&a, &calls](double value) {
        ++calls;
        return value + a.coeff(0);
    };
    const CoefficientWiseOperation hidden(b, callback);
    CHECK_FALSE(assignment_is_safe(hidden, a));
    const CoefficientWiseOperation stateless(b, [](double v) { return -v; });
    CHECK_FALSE(assignment_is_safe(stateless, a));
    const Operation hidden_binary(a, b, [&calls](double x, double y) {
        ++calls;
        return x + y;
    });
    CHECK_FALSE(assignment_is_safe(hidden_binary, a));
    const ScalarOperation hidden_scalar(b, 2.0, [&calls](double x, double y) {
        ++calls;
        return x * y;
    });
    CHECK_FALSE(assignment_is_safe(hidden_scalar, a));
    CHECK(calls == 0);
    CHECK_FALSE(assignment_is_safe(PretendLeaf{}, a));
    CHECK_FALSE(assignment_is_safe(a, PretendLeaf{}));
    CHECK_FALSE(assignment_is_safe(42, a));

    // Another owning array, of another element type, is still independent.
    MDArray<float, extents<4>> other_type;
    CHECK(assignment_is_safe(other_type, a));
    MDArray<OpaqueValue, extents<4>> opaque;
    CHECK_FALSE(assignment_is_safe(opaque, opaque));
    const CoefficientWiseOperation overloaded(opaque, std::negate<>{});
    CHECK_FALSE(assignment_is_safe(overloaded, opaque));
    STATIC_CHECK_FALSE((zipper::expression::detail::is_pointwise_operation<
                       std::negate<>, OpaqueValue>::value));
}

TEST_CASE("assignment_safety_rejects_nonpointwise_nodes", "[expression][assignment_safety]") {
    zipper::Matrix<double, 2, 2> a, b;
    const auto transpose = a.transpose();
    CHECK_FALSE(assignment_is_safe(transpose.expression(), a.expression()));
    const auto product = a * b;
    CHECK_FALSE(assignment_is_safe(product.expression(), a.expression()));
    const auto gathered = a.col_slice(std::array<zipper::index_type, 2>{1, 0});
    CHECK_FALSE(assignment_is_safe(gathered.expression(), a.expression()));
    const auto reduced = a.colwise().sum();
    zipper::Vector<double, 2> v;
    CHECK_FALSE(assignment_is_safe(reduced.expression(), v.expression()));
}
