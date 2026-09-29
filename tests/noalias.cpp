#include "catch_include.hpp"
#include <zipper/CSRMatrix.hpp>
#include <zipper/DataArray.hpp>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Identity.hpp>
#include <zipper/expression/nullary/Random.hpp>
#include <zipper/expression/unary/AntiSlice.hpp>
#include <zipper/expression/unary/ExtentView.hpp>
#include <zipper/expression/unary/Reshape.hpp>
#include <cstdlib>
#include <new>
#include <utility>

namespace {
bool count_allocations = false;
std::size_t allocations = 0;
}

// Scoped below to evaluation only, excluding Catch's own allocations.
auto operator new(std::size_t size) -> void * {
    if (void *ptr = std::malloc(size == 0 ? 1 : size)) {
        if (count_allocations) {
            ++allocations;
        }
        return ptr;
    }
    throw std::bad_alloc();
}
void operator delete(void *ptr) noexcept { std::free(ptr); }
void operator delete(void *ptr, std::size_t) noexcept { std::free(ptr); }
// Array forms: dynamic DenseData allocates via new[]. Some runtimes (e.g.
// ASan) do not route the default new[] through the replaced operator new.
auto operator new[](std::size_t size) -> void * { return operator new(size); }
void operator delete[](void *ptr) noexcept { std::free(ptr); }
void operator delete[](void *ptr, std::size_t) noexcept { std::free(ptr); }

namespace {
struct Counted {
    int value = 0;
    static inline int writes = 0;
    static inline int copies = 0;

    Counted() = default;
    explicit Counted(int v) : value(v) {}
    Counted(const Counted &other) : value(other.value) { ++copies; }
    auto operator=(const Counted &other) -> Counted & {
        value = other.value;
        ++writes;
        return *this;
    }
};

struct Measurement {
    Measurement() {
        allocations = 0;
        Counted::writes = 0;
        Counted::copies = 0;
        count_allocations = true;
    }
    ~Measurement() { count_allocations = false; }
};

template <typename Extents>
class CustomEvaluator;
class ScalarReader;
}

namespace zipper::expression::detail {
template <typename Extents>
struct ExpressionTraits<::CustomEvaluator<Extents>>
    : BasicExpressionTraits<Counted, Extents, AccessFeatures::mutable_value()> {
    using assign_strategy = FiberAssignStrategy<0>;
    static constexpr bool is_coefficient_consistent = false;
};

template <>
struct ExpressionTraits<::ScalarReader>
    : BasicExpressionTraits<int, zipper::extents<>, AccessFeatures::mutable_value()> {
    static constexpr bool is_coefficient_consistent = false;
};
}

namespace {
template <typename Extents>
class CustomEvaluator
    : public zipper::expression::ExpressionBase<CustomEvaluator<Extents>> {
public:
    CustomEvaluator(Extents shape, int &calls, int &reads)
        : m_extents(shape), m_calls(calls), m_reads(reads) {}
    auto extents() const -> Extents { return m_extents; }
    auto extent(zipper::rank_type i) const -> zipper::index_type {
        return m_extents.extent(i);
    }
    template <typename... Indices>
    auto coeff(Indices...) const -> Counted {
        ++m_reads;
        return Counted(7);
    }
    template <typename To>
    void assign_to(To &to) const {
        ++m_calls;
        zipper::utils::extents::for_each_index<zipper::detail::NoLayoutPreference>(
            to.extents(), [&](auto... idxs) { to(idxs...) = Counted(7); });
    }
private:
    Extents m_extents;
    int &m_calls;
    int &m_reads;
};

// Deliberately supports only (), not ignored index arguments. It may read a
// destination coefficient, so ordinary assignment must snapshot before writes.
class ScalarReader : public zipper::expression::ExpressionBase<ScalarReader> {
public:
    ScalarReader(const int &value, int &calls) : m_value(value), m_calls(calls) {}
    auto extents() const -> zipper::extents<> { return {}; }
    auto coeff() const -> int {
        ++m_calls;
        return m_value + 1;
    }
private:
    const int &m_value;
    int &m_calls;
};

template <typename T>
concept HasNoAlias = requires(T &&value) { std::forward<T>(value).noalias(); };

template <typename T>
concept HasNoAliasProxy = requires(T &value) {
    zipper::detail::NoAliasProxy<T>(value);
};

template <typename T>
concept DeducesNoAliasProxy = requires(T &value) {
    zipper::detail::NoAliasProxy(value);
};

template <typename From, typename To>
concept HasAssignHelper = requires {
    typename zipper::expression::detail::AssignHelper<From, To>;
};

template <typename To, typename From>
concept NoAliasAssignable = requires(To &to, const From &from) {
    { to.noalias() = from } -> std::same_as<To &>;
};
}

TEST_CASE("noalias_destination_contract", "[assignment][noalias]") {
    using namespace zipper;
    using Owned = Vector<int, 3>;
    using Expr = Owned::expression_type;
    STATIC_CHECK(HasNoAlias<Owned &>);
    STATIC_CHECK_FALSE(HasNoAlias<const Owned &>);
    STATIC_CHECK_FALSE(HasNoAlias<Owned>);
    STATIC_CHECK_FALSE(HasNoAlias<const Owned>);
    STATIC_CHECK(HasNoAlias<VectorBase<Expr &> &>);
    STATIC_CHECK_FALSE(HasNoAlias<VectorBase<const Expr &> &>);
    STATIC_CHECK_FALSE(HasNoAlias<VectorBase<const Expr> &>);
    STATIC_CHECK(HasNoAlias<Owned::span_type &>);
    STATIC_CHECK_FALSE(HasNoAlias<Owned::const_span_type &>);
    STATIC_CHECK_FALSE(HasNoAlias<COOMatrix<double, 3, 3> &>);
    STATIC_CHECK_FALSE(HasNoAlias<CSRMatrix<double, 3, 3> &>);
    STATIC_CHECK(NoAliasAssignable<Owned, Owned>);
    STATIC_CHECK(NoAliasAssignable<Owned, Expr>);
    STATIC_CHECK_FALSE(NoAliasAssignable<Owned, Vector<int, 2>>);
    STATIC_CHECK_FALSE(NoAliasAssignable<Owned, Matrix<int, 3, 3>>);
    STATIC_CHECK_FALSE(NoAliasAssignable<Owned, Vector<int, 2>::expression_type>);
    STATIC_CHECK_FALSE(NoAliasAssignable<Owned, Matrix<int, 3, 3>::expression_type>);
    STATIC_CHECK(HasNoAliasProxy<Owned>);
    STATIC_CHECK(HasNoAliasProxy<VectorBase<Expr &>>);
    STATIC_CHECK(HasNoAliasProxy<Owned::span_type>);
    STATIC_CHECK_FALSE(HasNoAliasProxy<const Owned>);
    STATIC_CHECK_FALSE(HasNoAliasProxy<VectorBase<const Expr &>>);
    STATIC_CHECK_FALSE(HasNoAliasProxy<VectorBase<const Expr>>);
    STATIC_CHECK_FALSE(HasNoAliasProxy<Owned::const_span_type>);
    STATIC_CHECK_FALSE(HasNoAliasProxy<COOMatrix<double, 3, 3>>);
    STATIC_CHECK_FALSE(HasNoAliasProxy<CSRMatrix<double, 3, 3>>);
    STATIC_CHECK_FALSE(HasNoAliasProxy<Expr>);
    STATIC_CHECK_FALSE(HasNoAliasProxy<int>);
    STATIC_CHECK(DeducesNoAliasProxy<Owned>);
    STATIC_CHECK_FALSE(DeducesNoAliasProxy<const Owned>);
    STATIC_CHECK_FALSE(DeducesNoAliasProxy<VectorBase<const Expr &>>);
    STATIC_CHECK_FALSE(DeducesNoAliasProxy<Owned::const_span_type>);
    STATIC_CHECK_FALSE(DeducesNoAliasProxy<COOMatrix<double, 3, 3>>);
    STATIC_CHECK_FALSE(DeducesNoAliasProxy<CSRMatrix<double, 3, 3>>);
    STATIC_CHECK(HasAssignHelper<Expr, Expr>);
    STATIC_CHECK(HasAssignHelper<expression::nullary::Constant<int>, Expr>);
    STATIC_CHECK_FALSE(HasAssignHelper<Expr, const Expr>);
    STATIC_CHECK_FALSE(HasAssignHelper<Expr, const Expr &>);
    STATIC_CHECK_FALSE(HasAssignHelper<Expr, Owned::const_span_type::expression_type>);
    STATIC_CHECK_FALSE(HasAssignHelper<Vector<int, 2>::expression_type, Expr>);
    STATIC_CHECK_FALSE(HasAssignHelper<Matrix<int, 3, 3>::expression_type, Expr>);

    Owned source{1, 2, 3};
    Owned target;
    CHECK(&(target.noalias() = source) == &target);
    CHECK(target == source);
    CHECK(&(target.noalias() = source.expression()) == &target);
    auto span = target.as_span();
    CHECK(&(span.noalias() = source * 2) == &span);
    CHECK(target == Owned{2, 4, 6});
}

TEST_CASE("independent_evaluation_writes_once", "[assignment][noalias]") {
    using namespace zipper;
    int calls = 0;
    int reads = 0;
    CustomEvaluator source(extents<dynamic_extent>(6), calls, reads);
    VectorX<Counted> target(6);
    std::size_t expected_allocations = 0;
    int expected_writes = 6;

    SECTION("conservative_assignment_snapshots") {
        Measurement measurement;
        target = source;
        expected_allocations = 1;
        expected_writes = 12;
    }
    SECTION("raw_noalias") {
        Measurement measurement;
        target.noalias() = source;
    }
    SECTION("wrapped_noalias") {
        VectorBase<const decltype(source) &> wrapped(std::in_place, source);
        Measurement measurement;
        target.noalias() = wrapped;
    }
    SECTION("span_noalias") {
        auto span = target.as_span();
        Measurement measurement;
        span.noalias() = source;
    }
    SECTION("resizing_noalias") {
        target.resize(1);
        Measurement measurement;
        target.noalias() = source;
    }
    SECTION("fresh_vector") {
        Measurement measurement;
        VectorX<Counted> result(source);
        target = std::move(result);
        expected_allocations = 1;
    }
    SECTION("eval_vector") {
        VectorBase<const decltype(source) &> wrapped(std::in_place, source);
        Measurement measurement;
        target = wrapped.eval();
        expected_allocations = 1;
    }
    CHECK(allocations == expected_allocations);
    CHECK(Counted::writes == expected_writes);
    CHECK(Counted::copies == 0);
    CHECK(calls == 1);
    CHECK(reads == 0);
    CHECK(target.extent(0) == 6);
    CHECK(target(5).value == 7);
}

TEST_CASE("fresh_matrix_and_data_array_evaluate_once", "[assignment][noalias]") {
    using namespace zipper;
    int calls = 0;
    int reads = 0;
    CustomEvaluator source(extents<dynamic_extent, dynamic_extent>(2, 3), calls, reads);
    SECTION("matrix_constructor") {
        Measurement measurement;
        MatrixXX<Counted> result(source);
    }
    SECTION("matrix_eval") {
        MatrixBase<const decltype(source) &> wrapped(std::in_place, source);
        Measurement measurement;
        auto result = wrapped.eval();
    }
    SECTION("data_array_constructor") {
        Measurement measurement;
        DataArray<Counted, dynamic_extent, dynamic_extent> result(source);
    }
    SECTION("data_array_eval") {
        DataArrayBase<const decltype(source) &> wrapped(std::in_place, source);
        Measurement measurement;
        auto result = wrapped.eval();
    }
    CHECK(allocations == 1);
    CHECK(Counted::writes == 6);
    CHECK(Counted::copies == 0);
    CHECK(calls == 1);
    CHECK(reads == 0);
}

TEST_CASE("span_make_owned_has_one_allocation_and_copy_pass", "[assignment][noalias]") {
    using namespace zipper;
    VectorX<Counted> source(6);
    auto span = source.as_span();
    VectorX<Counted>::expression_type owned;
    {
        Measurement measurement;
        owned = span.expression().make_owned();
    }
    CHECK(allocations == 1);
    CHECK(Counted::writes == 6);
    CHECK(owned.extent(0) == 6);
    source(0) = Counted(9);
    CHECK(owned(0).value == 0);
}

TEST_CASE("fresh_static_storage_has_no_snapshot", "[assignment][noalias]") {
    using namespace zipper;
    int calls = 0;
    int reads = 0;
    CustomEvaluator source(extents<2, 3>{}, calls, reads);
    {
        Measurement measurement;
        Matrix<Counted, 2, 3> result(source);
    }
    CHECK(allocations == 0);
    CHECK(Counted::writes == 6);
    CHECK(Counted::copies == 0);
    CHECK(calls == 1);
    CHECK(reads == 0);
}

TEST_CASE("noalias_shape_and_scalar_semantics", "[assignment][noalias]") {
    using namespace zipper;
    Matrix<int, 2, 3> source{{1, 2, 3}, {4, 5, 6}};
    MatrixXX<int> target(1, 1);
    target.noalias() = source.transpose();
    CHECK(target == Matrix<int, 3, 2>{{1, 4}, {2, 5}, {3, 6}});
    auto span = target.as_span();
    auto *data = target.expression().data();
    span.noalias() = expression::nullary::Constant<int>(4);
    target.noalias() = expression::nullary::Constant<int>(5);
    CHECK(target.extent(0) == 3);
    CHECK(target.extent(1) == 2);
    CHECK(target.expression().data() == data);
    CHECK(target == Matrix<int, 3, 2>{{5, 5}, {5, 5}, {5, 5}});

    int calls = 0;
    int value = 8;
    ScalarReader independent(value, calls);
    target.noalias() = independent;
    CHECK(calls == 1);
    CHECK(target == Matrix<int, 3, 2>{{9, 9}, {9, 9}, {9, 9}});
    calls = 0;
    ScalarReader aliased(target(0, 0), calls);
    target = aliased;
    CHECK(calls == 1);
    CHECK(target == Matrix<int, 3, 2>{{10, 10}, {10, 10}, {10, 10}});
    CHECK(target.expression().data() == data);

    DataArray<int> scalar;
    scalar.noalias() = independent;
    CHECK(scalar() == 9);
    Matrix<int, 2, 3> fresh_scalar(independent);
    CHECK(fresh_scalar == Matrix<int, 2, 3>{{9, 9, 9}, {9, 9, 9}});

    MatrixXX<int> incompatible(3, 2);
    CHECK_THROWS_AS(source.noalias() = incompatible, std::runtime_error);
    CHECK(source == Matrix<int, 2, 3>{{1, 2, 3}, {4, 5, 6}});

    target.noalias() = expression::nullary::Identity<int>{};
    CHECK(target == Matrix<int, 3, 2>{{1, 0}, {0, 1}, {0, 0}});
    Matrix<int, 2, 2> identity(expression::nullary::Identity<int>{});
    CHECK(identity == Matrix<int, 2, 2>{{1, 0}, {0, 1}});
}

TEST_CASE("noalias_partial_transform_retains_fiber_evaluator", "[assignment][noalias]") {
    using namespace zipper;
    Matrix<int, 2, 3> source{{1, 2, 3}, {4, 5, 6}};
    int calls = 0;
    auto transformed = source.colwise().transform([&](const auto &col) {
        ++calls;
        return 2 * col;
    });
    MatrixXX<int> target(1, 1);
    target.noalias() = transformed;
    CHECK(calls == 3);
    CHECK(target == Matrix<int, 2, 3>{{2, 4, 6}, {8, 10, 12}});
}

TEST_CASE("rank_zero_scalar_callback_snapshots_once", "[assignment][rank_zero]") {
    using namespace zipper;
    Vector<int, 2> a{1, 2};
    const auto &external = a;
    expression::nullary::Constant<int> seed(0);
    int calls = 0;
    const expression::unary::CoefficientWiseOperation source(seed, [&](int) {
        ++calls;
        return external(0) + 1;
    });
    STATIC_CHECK(decltype(source)::traits::is_coefficient_consistent);
    SECTION("ordinary_captured_alias") { a = source; }
    SECTION("fresh") { a = Vector<int, 2>(source); }
    SECTION("noalias_independent_destination") {
        Vector<int, 2> result;
        result.noalias() = source;
        a = result;
    }
    CHECK(a == Vector<int, 2>{2, 2});
    CHECK(calls == 1);
}

TEST_CASE("rank_zero_coeff_only_scalar_cardinality", "[assignment][rank_zero]") {
    using namespace zipper;
    int calls = 0;
    const int value = 8;
    const ScalarReader source(value, calls);
    Matrix<int, 2, 2> target;
    SECTION("ordinary") { target = source; }
    SECTION("fresh") { target = Matrix<int, 2, 2>(source); }
    SECTION("noalias") { target.noalias() = source; }
    SECTION("unary_forwarding_signature") {
        const expression::unary::CoefficientWiseOperation wrapped(
            source, [](int v) { return v; });
        target.noalias() = wrapped;
    }
    CHECK(calls == 1);
    CHECK(target == Matrix<int, 2, 2>{{9, 9}, {9, 9}});
}

TEST_CASE("rank_zero_identity_plus_scalar_is_indexed", "[assignment][rank_zero]") {
    using namespace zipper;
    expression::nullary::Identity<int> identity;
    DataArray<int> scalar;
    scalar() = 2;
    const expression::binary::Operation source(
        identity, scalar.expression(), std::plus<int>{});
    STATIC_CHECK_FALSE(decltype(source)::traits::is_coefficient_consistent);
    MatrixXX<int> target(2, 2);
    auto *data = target.expression().data();
    SECTION("ordinary") { target = source; }
    SECTION("noalias") { target.noalias() = source; }
    SECTION("fresh") {
        Matrix<int, 2, 2> fresh(source);
        target = fresh;
    }
    SECTION("scalar_aliases_destination") {
        target = expression::nullary::Constant<int>(2);
        using ScalarSpan = expression::nullary::MDSpan<int, extents<>>;
        const ScalarSpan aliased(std::span<int>(data, 1));
        const expression::binary::Operation dependent(identity, aliased, std::plus<int>{});
        target = dependent;
    }
    SECTION("callback_reads_destination_through_const_reference") {
        target = expression::nullary::Constant<int>(2);
        const auto &external = target;
        int calls = 0;
        const expression::unary::CoefficientWiseOperation dependent(
            identity, [&](int v) {
                ++calls;
                return v + external(0, 0);
            });
        target = dependent;
        CHECK(calls == 4);
    }
    CHECK(target.expression().data() == data);
    CHECK(target == Matrix<int, 2, 2>{{3, 2}, {2, 3}});
}

TEST_CASE("rank_zero_random_arithmetic_remains_per_coefficient",
          "[assignment][rank_zero]") {
    using namespace zipper;
    auto random = expression::nullary::uniform_random<double>(
        extents<>{}, 0.0, 1.0, std::default_random_engine{123});
    auto expected = random;
    int calls = 0;
    const expression::unary::CoefficientWiseOperation unary(random, [&](double v) {
        ++calls;
        return v * 2;
    });
    DataArray<double> scalar;
    scalar() = 1;
    const expression::binary::Operation source(unary, scalar.expression(), std::plus<double>{});
    Matrix<double, 2, 2> target;
    SECTION("ordinary") { target = source; }
    SECTION("noalias") { target.noalias() = source; }
    SECTION("fresh") { target = Matrix<double, 2, 2>(source); }
    CHECK(calls == 4);
    for (index_type i = 0; i < 2; ++i) {
        for (index_type j = 0; j < 2; ++j) {
            CHECK(target(i, j) == expected() * 2 + 1);
        }
    }
}

TEST_CASE("rank_zero_custom_evaluator_uses_destination_shape",
          "[assignment][rank_zero][noalias]") {
    using namespace zipper;
    int calls = 0;
    int reads = 0;
    const CustomEvaluator source(extents<>{}, calls, reads);
    Matrix<Counted, 2, 3> target;
    int expected_writes = 6;
    SECTION("ordinary") {
        Measurement measurement;
        target = source;
        expected_writes = 12;
    }
    SECTION("noalias") {
        Measurement measurement;
        target.noalias() = source;
    }
    SECTION("fresh") {
        Measurement measurement;
        Matrix<Counted, 2, 3> fresh(source);
    }
    CHECK(allocations == 0);
    CHECK(Counted::writes == expected_writes);
    CHECK(Counted::copies == 0);
    CHECK(calls == 1);
    CHECK(reads == 0);
}

TEST_CASE("proven_pointwise_assignment_has_no_allocation", "[assignment][alias_safety]") {
    using namespace zipper;
    MatrixXX<int> a(Matrix<int, 2, 3>{{1, 2, 3}, {4, 5, 6}});
    MatrixXX<int> b(Matrix<int, 2, 3>{{6, 5, 4}, {3, 2, 1}});
    Matrix<int, 2, 3> expected;
    const auto *data = a.expression().data();
    SECTION("scalar_multiply") {
        expected = Matrix<int, 2, 3>{{2, 4, 6}, {8, 10, 12}};
        Measurement measurement;
        a = a * 2;
    }
    SECTION("binary_sum") {
        expected = Matrix<int, 2, 3>{{7, 7, 7}, {7, 7, 7}};
        Measurement measurement;
        a = a + b;
    }
    SECTION("negate") {
        expected = Matrix<int, 2, 3>{{-1, -2, -3}, {-4, -5, -6}};
        Measurement measurement;
        a = -a;
    }
    SECTION("distinct_same_shape_dense") {
        expected = b;
        // Use expression assignment, not the owning container's copy assignment.
        Measurement measurement;
        a.expression().assign(b.expression());
    }
    CHECK(allocations == 0);
    CHECK(a.expression().data() == data);
    CHECK(a == expected);
}

TEST_CASE("unproven_assignment_snapshots_before_mutation", "[assignment][alias_safety]") {
    using namespace zipper;
    using expression::detail::assignment_is_safe;
    SECTION("shifted_spans") {
        VectorX<int> values(Vector<int, 5>{1, 2, 3, 4, 5});
        using Span = expression::nullary::MDSpan<int, dextents<1>>;
        Span source(std::span<int>(values.expression().data(), 4));
        Span destination(std::span<int>(values.expression().data() + 1, 4));
        CHECK_FALSE(assignment_is_safe(source, destination));
        {
            Measurement measurement;
            destination.assign(source);
        }
        CHECK(allocations == 1);
        CHECK(values == Vector<int, 5>{1, 1, 2, 3, 4});
    }
    SECTION("transposed_dense_mapping") {
        MatrixXX<int> a(Matrix<int, 2, 2>{{1, 2}, {3, 4}});
        using Span = expression::nullary::MDSpan<int, dextents<2>, storage::layout_left>;
        Span source(typename Span::linear_accessor_type(
                        std::span<int>(a.expression().data(), 4)), dextents<2>(2, 2));
        CHECK_FALSE(assignment_is_safe(source, a.expression()));
        {
            Measurement measurement;
            a = source;
        }
        CHECK(allocations == 1);
        CHECK(a == Matrix<int, 2, 2>{{1, 3}, {2, 4}});
    }
    SECTION("captured_alias_in_callback") {
        VectorX<int> a(Vector<int, 4>{1, 2, 3, 4});
        VectorX<int> b(Vector<int, 4>{10, 20, 30, 40});
        int calls = 0;
        const expression::unary::CoefficientWiseOperation source(
            b.expression(), [&](int value) {
                ++calls;
                return value + a(0);
            });
        CHECK_FALSE(assignment_is_safe(source, a.expression()));
        {
            Measurement measurement;
            a = source;
        }
        CHECK(allocations == 1);
        CHECK(calls == 4);
        CHECK(a == Vector<int, 4>{11, 21, 31, 41});
    }
    SECTION("shape_changing_self_transpose") {
        MatrixXX<int> a(Matrix<int, 2, 3>{{1, 2, 3}, {4, 5, 6}});
        const auto source = a.transpose();
        CHECK_FALSE(assignment_is_safe(source.expression(), a.expression()));
        {
            Measurement measurement;
            a = source;
        }
        CHECK(allocations == 1);
        CHECK(a == Matrix<int, 3, 2>{{1, 4}, {2, 5}, {3, 6}});
    }
}

TEST_CASE("noalias_slice_destination_contract", "[assignment][noalias][slice]") {
    using namespace zipper;
    using Buffer = Matrix<int, 4, dynamic_extent, false>;
    using Columns = decltype(zipper::slice(index_type{0}, index_type{0}));
    using Rows2 = decltype(zipper::slice(
        index_type{0}, std::integral_constant<index_type, 2>{}));
    using Block = decltype(std::declval<Buffer &>()(full_extent_t{},
                                                    std::declval<Columns>()));
    using ConstBlock = decltype(std::declval<const Buffer &>()(
        full_extent_t{}, std::declval<Columns>()));
    using NestedBlock = decltype(std::declval<Block &>()(
        std::declval<Rows2>(), full_extent_t{}));
    using SpanBlock = decltype(std::declval<Buffer::span_type &>()(
        full_extent_t{}, std::declval<Columns>()));
    using ConstSpanBlock = decltype(std::declval<Buffer::const_span_type &>()(
        full_extent_t{}, std::declval<Columns>()));

    STATIC_CHECK(HasNoAlias<Block &>);
    STATIC_CHECK(HasNoAlias<NestedBlock &>);
    STATIC_CHECK(HasNoAlias<SpanBlock &>);
    STATIC_CHECK_FALSE(HasNoAlias<const Block &>);
    STATIC_CHECK_FALSE(HasNoAlias<ConstBlock &>);
    STATIC_CHECK_FALSE(HasNoAlias<ConstSpanBlock &>);
    STATIC_CHECK(NoAliasAssignable<Block, Matrix<int, 4, 3>>);
    STATIC_CHECK_FALSE(NoAliasAssignable<Block, Matrix<int, 3, 3>>);
}

TEST_CASE("noalias_slice_packs_without_snapshot", "[assignment][noalias][slice]") {
    using namespace zipper;
    // Pack a 4-row strip of a column-major source into one panel of a larger
    // buffer, as GEMM packing does. Both slices have a dynamic column extent.
    Matrix<int, dynamic_extent, dynamic_extent, false> source(6, 5);
    for (index_type j = 0; j < 5; ++j) {
        for (index_type i = 0; i < 6; ++i) {
            source(i, j) = static_cast<int>(10 * i + j);
        }
    }
    Matrix<int, 4, dynamic_extent, false> buffer(4, 10);
    auto panel = buffer(full_extent_t{}, zipper::slice(index_type{5}, index_type{5}));
    const auto strip = source(
        zipper::slice(index_type{1}, std::integral_constant<index_type, 4>{}),
        full_extent_t{});

    bool conservative = false;
    SECTION("conservative_assignment_snapshots") {
        Measurement measurement;
        panel = strip;
        conservative = true;
    }
    SECTION("noalias") {
        Measurement measurement;
        CHECK(&(panel.noalias() = strip) == &panel);
    }
    if (conservative) {
        CHECK(allocations > 0);
    } else {
        CHECK(allocations == 0);
    }
    for (index_type p = 0; p < 5; ++p) {
        for (index_type r = 0; r < 4; ++r) {
            CHECK(buffer(r, p) == 0);                  // untouched panel
            CHECK(buffer(r, 5 + p) == source(1 + r, p)); // packed panel
        }
    }
}

TEST_CASE("noalias_nested_slice_writes_through", "[assignment][noalias][slice]") {
    using namespace zipper;
    Matrix<int, dynamic_extent, dynamic_extent, false> target(4, 4);
    auto block = target(zipper::slice(index_type{1}, index_type{3}),
                        zipper::slice(index_type{1}, index_type{3}));
    auto inner = block(zipper::slice(index_type{1}, index_type{2}), full_extent_t{});
    {
        Measurement measurement;
        inner.noalias() = Matrix<int, 2, 3>{{1, 2, 3}, {4, 5, 6}};
    }
    CHECK(allocations == 0);
    CHECK(target == Matrix<int, 4, 4>{{0, 0, 0, 0},
                                      {0, 0, 0, 0},
                                      {0, 1, 2, 3},
                                      {0, 4, 5, 6}});
}

TEST_CASE("noalias_slice_uses_custom_evaluator_once", "[assignment][noalias][slice]") {
    using namespace zipper;
    int calls = 0;
    int reads = 0;
    CustomEvaluator source(extents<dynamic_extent>(4), calls, reads);
    VectorX<Counted> target(6);
    auto middle = target.segment(1, 4);
    {
        Measurement measurement;
        middle.noalias() = source;
    }
    CHECK(allocations == 0);
    CHECK(Counted::writes == 4);
    CHECK(Counted::copies == 0);
    CHECK(calls == 1);
    CHECK(reads == 0);
    CHECK(target(0).value == 0);
    CHECK(target(1).value == 7);
    CHECK(target(4).value == 7);
    CHECK(target(5).value == 0);
}

TEST_CASE("noalias_forwarding_view_contract", "[assignment][noalias][views]") {
    using namespace zipper;
    namespace unary = expression::unary;
    using Dense = Matrix<int, dynamic_extent, dynamic_extent, false>;
    using Expr = Dense::expression_type;
    using Columns = decltype(zipper::slice(index_type{0}, index_type{0}));
    using Transposed = decltype(std::declval<Dense &>().transpose());
    using Swizzled = decltype(std::declval<Dense &>().template swizzle<1, 0>());
    using Diagonal = decltype(std::declval<Dense &>().diagonal());
    using Ref = decltype(std::declval<Dense &>().ref());
    using SliceOfTranspose = decltype(std::declval<Transposed &>()(
        full_extent_t{}, std::declval<Columns>()));
    using TransposeOfSlice = decltype(std::declval<Dense &>()(
        full_extent_t{}, std::declval<Columns>()).transpose());
    using Reshaped = MatrixBase<unary::Reshape<Expr &, dextents<2>>>;
    using Retyped = MatrixBase<unary::ExtentView<Expr &, dextents<2>>>;
    using Column = MatrixBase<unary::AntiSlice<VectorX<int>::expression_type &, 1>>;

    STATIC_CHECK(HasNoAlias<Transposed &>);
    STATIC_CHECK(HasNoAlias<Swizzled &>);
    STATIC_CHECK(HasNoAlias<Diagonal &>);
    STATIC_CHECK(HasNoAlias<Ref &>);
    STATIC_CHECK(HasNoAlias<SliceOfTranspose &>);
    STATIC_CHECK(HasNoAlias<TransposeOfSlice &>);
    STATIC_CHECK(HasNoAlias<Reshaped &>);
    STATIC_CHECK(HasNoAlias<Retyped &>);
    STATIC_CHECK(HasNoAlias<Column &>);

    STATIC_CHECK_FALSE(HasNoAlias<decltype(std::declval<const Dense &>().transpose()) &>);
    STATIC_CHECK_FALSE(HasNoAlias<decltype(std::declval<const Dense &>().diagonal()) &>);
    STATIC_CHECK_FALSE(HasNoAlias<decltype(std::declval<const Dense &>().ref()) &>);
    STATIC_CHECK_FALSE(HasNoAlias<MatrixBase<unary::Reshape<const Expr &, dextents<2>>> &>);
    STATIC_CHECK_FALSE(HasNoAlias<MatrixBase<unary::ExtentView<const Expr &, dextents<2>>> &>);
    STATIC_CHECK_FALSE(HasNoAlias<const Transposed &>);
}

TEST_CASE("noalias_forwarding_views_write_through", "[assignment][noalias][views]") {
    using namespace zipper;
    namespace unary = expression::unary;
    using Dense = Matrix<int, dynamic_extent, dynamic_extent, false>;
    Dense target(3, 3);
    const Matrix<int, 3, 3> source{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    SECTION("transpose") {
        auto view = target.transpose();
        {
            Measurement measurement;
            CHECK(&(view.noalias() = source) == &view);
        }
        CHECK(target == Matrix<int, 3, 3>{{1, 4, 7}, {2, 5, 8}, {3, 6, 9}});
    }
    SECTION("diagonal") {
        auto view = target.diagonal();
        {
            Measurement measurement;
            view.noalias() = Vector<int, 3>{1, 2, 3};
        }
        CHECK(target == Matrix<int, 3, 3>{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}});
    }
    SECTION("slice_of_transpose") {
        auto transposed = target.transpose();
        auto view = transposed(full_extent_t{}, zipper::slice(index_type{1}, index_type{2}));
        {
            Measurement measurement;
            view.noalias() = Matrix<int, 3, 2>{{1, 2}, {3, 4}, {5, 6}};
        }
        CHECK(target == Matrix<int, 3, 3>{{0, 0, 0}, {1, 3, 5}, {2, 4, 6}});
    }
    SECTION("transpose_of_slice") {
        auto view = target(full_extent_t{}, zipper::slice(index_type{1}, index_type{2}))
                        .transpose();
        {
            Measurement measurement;
            view.noalias() = Matrix<int, 2, 3>{{1, 2, 3}, {4, 5, 6}};
        }
        CHECK(target == Matrix<int, 3, 3>{{0, 1, 4}, {0, 2, 5}, {0, 3, 6}});
    }
    SECTION("ref") {
        auto view = target.ref();
        {
            Measurement measurement;
            view.noalias() = source;
        }
        CHECK(target == source);
    }
    SECTION("reshape") {
        MatrixBase<unary::Reshape<Dense::expression_type &, dextents<2>>> view(
            std::in_place, target.expression(), dextents<2>(1, 9));
        {
            Measurement measurement;
            view.noalias() = Matrix<int, 1, 9>{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
        }
        // Reshape unravels row-major into the child's (row, column) indices.
        CHECK(target == source);
    }
    SECTION("extent_view") {
        MatrixBase<unary::ExtentView<Dense::expression_type &, extents<3, 3>>> view(
            std::in_place, target.expression());
        {
            Measurement measurement;
            view.noalias() = source;
        }
        CHECK(target == source);
    }
    SECTION("antislice") {
        VectorX<int> column(3);
        MatrixBase<unary::AntiSlice<VectorX<int>::expression_type &, 1>> view(
            std::in_place, column.expression());
        {
            Measurement measurement;
            view.noalias() = Matrix<int, 3, 1>{{1}, {2}, {3}};
        }
        CHECK(column == Vector<int, 3>{1, 2, 3});
    }
    CHECK(allocations == 0);
}
