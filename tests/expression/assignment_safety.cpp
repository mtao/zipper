// Intentionally first: this header must also work from inside AssignHelper,
// before MDArray/MDSpan or any constant template is defined.
#include <zipper/expression/detail/AssignmentSafety.hpp>

#include "catch_include.hpp"
#include <zipper/Matrix.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/MDArray.hpp>
#include <zipper/expression/nullary/MDSpan.hpp>
#include <zipper/expression/nullary/StaticConstant.hpp>

namespace {
using zipper::expression::detail::assignment_is_safe;
using zipper::expression::nullary::MDArray;
using zipper::expression::nullary::MDSpan;
using zipper::expression::nullary::Constant;
using zipper::expression::nullary::StaticConstant;
using zipper::expression::unary::CoefficientWiseOperation;
using zipper::expression::unary::ScalarOperation;
using zipper::expression::binary::Operation;
using zipper::extents;

struct PureNegate {
    auto operator()(double value) const -> double { return -value; }
};

struct CustomAccessor : zipper::default_accessor_policy<double> {};

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

TEST_CASE("assignment_safety_overlapping_spans", "[expression][assignment_safety]") {
    std::array<double, 10> data{};
    using Span = MDSpan<double, extents<4>>;
    Span a(std::span<double, 4>(data.data(), 4));
    Span same(std::span<double, 4>(data.data(), 4));
    Span shifted(std::span<double, 4>(data.data() + 1, 4));
    Span partial(std::span<double, 4>(data.data() + 3, 4));
    Span adjacent(std::span<double, 4>(data.data() + 4, 4));
    MDSpan<const double, extents<4>> read(std::span<const double, 4>(data.data(), 4));
    CHECK(assignment_is_safe(same, a));
    CHECK(assignment_is_safe(read, a));
    CHECK_FALSE(assignment_is_safe(a, read));
    MDSpan<const double, extents<4>> independent_read(
        std::span<const double, 4>(data.data() + 4, 4));
    CHECK(assignment_is_safe(independent_read, a));
    CHECK_FALSE(assignment_is_safe(shifted, a));
    CHECK_FALSE(assignment_is_safe(a, shifted));
    CHECK_FALSE(assignment_is_safe(partial, a));
    CHECK_FALSE(assignment_is_safe(a, partial));
    CHECK(assignment_is_safe(adjacent, a));
    CHECK(assignment_is_safe(a, adjacent));
    const Operation unsafe_sum(a, shifted, std::plus<>{});
    CHECK_FALSE(assignment_is_safe(unsafe_sum, a));
    const CoefficientWiseOperation unsafe_neg(unsafe_sum, std::negate<>{});
    CHECK_FALSE(assignment_is_safe(unsafe_neg, a));
    const Operation safe_sum(read, adjacent, std::plus<>{});
    CHECK(assignment_is_safe(safe_sum, a));
}

TEST_CASE("assignment_safety_mapping_and_backing_span", "[expression][assignment_safety]") {
    std::array<double, 4> data{};
    MDSpan<double, extents<2, 2>, zipper::storage::layout_right> right(data);
    MDSpan<double, extents<2, 2>, zipper::storage::layout_left> left(data);
    CHECK_FALSE(assignment_is_safe(left, right));
    CHECK_FALSE(assignment_is_safe(right, left));
    CHECK(assignment_is_safe(left, left));

    // A malformed logical shape must not allow out-of-bounds end arithmetic.
    using DynamicSpan = MDSpan<double, zipper::dextents<2>>;
    DynamicSpan short_span(typename DynamicSpan::linear_accessor_type(data),
                           zipper::dextents<2>(2, 3));
    MDArray<double, zipper::dextents<2>> destination(zipper::dextents<2>(2, 3));
    CHECK_FALSE(assignment_is_safe(short_span, destination));
    CHECK_FALSE(assignment_is_safe(destination, short_span));

    MDSpan<double, extents<4>, zipper::storage::layout_right, CustomAccessor> custom(data);
    MDSpan<double, extents<4>> plain(data);
    CHECK_FALSE(assignment_is_safe(custom, plain));
    CHECK_FALSE(assignment_is_safe(plain, custom));
}

TEST_CASE("assignment_safety_shape_and_constants", "[expression][assignment_safety]") {
    MDArray<double, zipper::dextents<2>> a(zipper::dextents<2>(2, 3));
    MDArray<double, zipper::dextents<2>> b(zipper::dextents<2>(3, 2));
    const auto *before = a.data();
    CHECK_FALSE(assignment_is_safe(b, a));
    CHECK(a.extents() == zipper::dextents<2>(2, 3));
    CHECK(a.data() == before);
    CHECK(assignment_is_safe(Constant<double>(4.0), a));
    CHECK(assignment_is_safe(Constant<double, 2, 3>(4.0), a));
    CHECK_FALSE(assignment_is_safe(Constant<double, 3, 2>(4.0), a));
    CHECK(assignment_is_safe(StaticConstant<double, 0>{}, a));
    CHECK(assignment_is_safe(StaticConstant<double, 1, 2, 3>{}, a));
    CHECK_FALSE(assignment_is_safe(StaticConstant<double, 1, 3, 2>{}, a));
    const Constant<double> scalar(2.0);
    const Operation sum(a, scalar, std::plus<>{});
    CHECK(assignment_is_safe(sum, a));

    MDArray<double, extents<>> rank_zero;
    CHECK(assignment_is_safe(rank_zero, rank_zero));
    CHECK_FALSE(assignment_is_safe(rank_zero, a));
    MDArray<double, zipper::dextents<1>> empty(zipper::dextents<1>(0));
    CHECK(assignment_is_safe(empty, empty));
    MDSpan<double, zipper::dextents<1>> empty_span(std::span<double>{});
    CHECK(assignment_is_safe(empty_span, empty));
}

TEST_CASE("assignment_safety_strided_storage", "[expression][assignment_safety]") {
#if defined(__cpp_lib_mdspan)
    using Layout = std::layout_stride;
#else
    using Layout = MDSPAN_IMPL_STANDARD_NAMESPACE::layout_stride;
#endif
    using Span = MDSpan<double, zipper::dextents<1>, Layout>;
    STATIC_CHECK(zipper::expression::detail::assignment_safety::DenseLeaf<Span>::value);
    // Mapping constructors are supplied by the integrated mapping branch. Keep
    // this test ready for that branch without importing any mapping factories.
    []<typename S>() {
        using Storage = typename S::linear_accessor_type;
        using Mapping = typename S::mapping_type;
        if constexpr (std::is_constructible_v<S, const Storage &, const Mapping &>) {
            std::array<double, 12> data{};
            const Mapping mapping(zipper::dextents<1>(3),
                                  std::array<zipper::index_type, 1>{2});
            S a(Storage(std::span<double>(data.data(), 5)), mapping);
            S same(Storage(std::span<double>(data.data(), 5)), mapping);
            S interleaved(Storage(std::span<double>(data.data() + 1, 5)), mapping);
            S independent(Storage(std::span<double>(data.data() + 6, 5)), mapping);
            CHECK(assignment_is_safe(same, a));
            CHECK(assignment_is_safe(independent, a));
            // Bounding spans include holes: conservative false negatives are OK.
            CHECK_FALSE(assignment_is_safe(interleaved, a));
            CHECK_FALSE(assignment_is_safe(a, interleaved));
        }
    }.template operator()<Span>();
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

    MDArray<float, extents<4>> other_type;
    CHECK_FALSE(assignment_is_safe(other_type, a));
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
