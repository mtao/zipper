#include "../../catch_include.hpp"
#include <memory>
#include <type_traits>
#include <utility>
#include <zipper/Vector.hpp>
#include <zipper/expression/unary/CoefficientWiseOperation.hpp>
#include <zipper/expression/unary/ScalarOperation.hpp>

namespace {

using Vector = zipper::Vector<int, 3>;
using Child = Vector::expression_type;
namespace unary = zipper::expression::unary;

template <typename V, typename Op>
concept HasUnaryExpr = requires(V &&v, const Op &op) {
    std::forward<V>(v).unary_expr(op);
};

template <typename Op>
concept HasCoefficientWiseOperation = requires(const Child &child, const Op &op) {
    unary::CoefficientWiseOperation(child, op);
};

struct Scalar {
    int value;
};

template <typename Op, bool OnRight>
concept HasScalarOperation = requires {
    typename unary::ScalarOperation<const Child &, Op, Scalar, OnRight>;
};

template <typename Op>
concept HasRightScalarConstructor = requires(const Child &child,
                                              const Scalar &scalar,
                                              const Op &op) {
    unary::ScalarOperation(child, scalar, op);
};

template <typename Op>
concept HasLeftScalarConstructor = requires(const Child &child,
                                             const Scalar &scalar,
                                             const Op &op) {
    unary::ScalarOperation(scalar, child, op);
};

struct MutableOnly {
    auto operator()(const int &) -> int;
    auto operator()(const int &, const Scalar &) -> int;
};

struct RvalueOnly {
    auto operator()(const int &) && -> int;
};

struct MutableInput {
    auto operator()(int &) const -> int;
};

struct RvalueInput {
    auto operator()(int &&) const -> int;
};

struct Overloaded {
    int offset;

    auto operator()(const int &v) const & -> long { return v + offset; }
    auto operator()(const int &) & -> short;
    auto operator()(int &&) const & -> double;
    auto operator()(const int &) && -> float;
};

struct Ordered {
    int offset;

    auto operator()(const int &v, const Scalar &s) const & -> long {
        return v - s.value + offset;
    }
    auto operator()(const Scalar &s, const int &v) const & -> double {
        return s.value - v + offset + 0.5;
    }
    auto operator()(const int &, const Scalar &) & -> short;
    auto operator()(int &&, Scalar &&) && -> float;
};

struct LeftOnly {
    auto operator()(const Scalar &s, const int &v) const & -> int {
        return s.value - v;
    }
};

struct RightOnly {
    auto operator()(const int &v, const Scalar &s) const & -> int {
        return v - s.value;
    }
};

struct MutableScalar {
    auto operator()(const int &, Scalar &) const -> int;
};

struct RvalueScalar {
    auto operator()(const int &, Scalar &&) const -> int;
};

struct ReferenceResult {
    auto operator()(const int &v) const & -> const int & { return v; }
    auto operator()(const int &v, const Scalar &) const & -> const int & {
        return v;
    }
    auto operator()(const Scalar &, const int &v) const & -> const int & {
        return v;
    }
    auto operator()(const int &, const Scalar &) & -> short;
    auto operator()(const Scalar &, const int &) & -> short;
    auto operator()(int &&, Scalar &&) && -> float;
    auto operator()(Scalar &&, int &&) && -> float;
};

struct VoidResult {
    auto operator()(const int &) const -> void;
    auto operator()(const int &, const Scalar &) const -> void;
    auto operator()(const Scalar &, const int &) const -> void;
};

struct MoveOnlyResult : ReferenceResult {
    MoveOnlyResult() = default;
    MoveOnlyResult(const MoveOnlyResult &) = delete;
    MoveOnlyResult(MoveOnlyResult &&) = default;
};

using DefaultUnary = unary::CoefficientWiseOperation<const Child &, ReferenceResult>;
using DefaultRight = unary::ScalarOperation<const Child &, ReferenceResult, Scalar, true>;
using DefaultLeft = unary::ScalarOperation<const Child &, ReferenceResult, Scalar, false>;
static_assert(std::is_constructible_v<DefaultUnary, const Child &>);
static_assert(std::is_copy_constructible_v<DefaultUnary>);
static_assert(std::is_move_constructible_v<DefaultUnary>);
static_assert(!std::is_constructible_v<DefaultUnary, int>);
static_assert(std::is_constructible_v<DefaultRight, const Child &, const Scalar &>);
static_assert(std::is_constructible_v<DefaultLeft, const Scalar &, const Child &>);
static_assert(!std::is_constructible_v<DefaultRight, const Scalar &, const Child &>);
static_assert(!std::is_constructible_v<DefaultLeft, const Child &, const Scalar &>);
static_assert(!std::is_constructible_v<DefaultRight, int, const Scalar &>);
static_assert(!std::is_constructible_v<DefaultLeft, const Scalar &, int>);

static_assert(std::default_initializable<MoveOnlyResult>);
static_assert(!HasCoefficientWiseOperation<MoveOnlyResult>);
static_assert(!HasRightScalarConstructor<MoveOnlyResult>);
static_assert(!HasLeftScalarConstructor<MoveOnlyResult>);
static_assert(!std::is_constructible_v<
              unary::CoefficientWiseOperation<const Child &, MoveOnlyResult>,
              const Child &>);
static_assert(!std::is_constructible_v<
              unary::ScalarOperation<const Child &, MoveOnlyResult, Scalar, true>,
              const Child &, const Scalar &>);
static_assert(!std::is_constructible_v<
              unary::ScalarOperation<const Child &, MoveOnlyResult, Scalar, false>,
              const Scalar &, const Child &>);

static_assert(std::is_invocable_v<const VoidResult &, const int &>);
static_assert(std::is_invocable_v<const VoidResult &, const int &, const Scalar &>);
static_assert(!unary::concepts::ScalarOperation<int, VoidResult>);
static_assert(!HasUnaryExpr<Vector &, VoidResult>);
static_assert(!HasCoefficientWiseOperation<VoidResult>);
static_assert(!HasScalarOperation<VoidResult, true>);
static_assert(!HasScalarOperation<VoidResult, false>);
static_assert(!HasRightScalarConstructor<VoidResult>);
static_assert(!HasLeftScalarConstructor<VoidResult>);

static_assert(std::is_invocable_v<MutableOnly &, const int &>);
static_assert(!std::is_invocable_v<const MutableOnly &, const int &>);
static_assert(std::is_invocable_v<RvalueOnly, const int &>);
static_assert(!std::is_invocable_v<const RvalueOnly &, const int &>);
static_assert(std::is_invocable_v<const MutableInput &, int &>);
static_assert(std::is_invocable_v<const RvalueInput &, int &&>);
static_assert(!HasUnaryExpr<Vector &, MutableOnly>);
static_assert(!HasUnaryExpr<Vector &, RvalueOnly>);
static_assert(!HasUnaryExpr<Vector &, MutableInput>);
static_assert(!HasUnaryExpr<Vector &, RvalueInput>);
static_assert(!HasCoefficientWiseOperation<MutableOnly>);
static_assert(!HasCoefficientWiseOperation<RvalueOnly>);
static_assert(!HasCoefficientWiseOperation<MutableInput>);
static_assert(!HasCoefficientWiseOperation<RvalueInput>);
static_assert(std::is_same_v<std::invoke_result_t<const Overloaded &, const int &>, long>);
static_assert(HasUnaryExpr<Vector &, Overloaded>);
static_assert(HasUnaryExpr<const Vector &, Overloaded>);
static_assert(HasUnaryExpr<Vector, Overloaded>);
static_assert(HasCoefficientWiseOperation<Overloaded>);
static_assert(!HasScalarOperation<MutableOnly, true>);
static_assert(!HasRightScalarConstructor<MutableOnly>);
static_assert(!HasScalarOperation<MutableScalar, true>);
static_assert(!HasRightScalarConstructor<MutableScalar>);
static_assert(!HasScalarOperation<RvalueScalar, true>);
static_assert(!HasRightScalarConstructor<RvalueScalar>);
static_assert(HasScalarOperation<LeftOnly, false>);
static_assert(!HasScalarOperation<LeftOnly, true>);
static_assert(HasLeftScalarConstructor<LeftOnly>);
static_assert(!HasRightScalarConstructor<LeftOnly>);
static_assert(HasScalarOperation<RightOnly, true>);
static_assert(!HasScalarOperation<RightOnly, false>);
static_assert(HasRightScalarConstructor<RightOnly>);
static_assert(!HasLeftScalarConstructor<RightOnly>);

} // namespace

TEST_CASE("callback_captures_and_construction", "[unary][callback]") {
    Vector v{1, 2, 3};
    CHECK(DefaultUnary(v.expression())(0) == 1);
    CHECK(DefaultRight(v.expression(), Scalar{7})(1) == 2);
    CHECK(DefaultLeft(Scalar{7}, v.expression())(2) == 3);
    int offset = 7;
    auto op = [offset](const int &value) { return value + offset; };
    using Op = decltype(op);
    using Node = unary::CoefficientWiseOperation<const Child &, Op>;
    static_assert(!std::is_default_constructible_v<Op>);
    static_assert(std::is_invocable_v<const Op &, const int &>);
    static_assert(unary::concepts::ScalarOperation<int, Op>);
    static_assert(HasUnaryExpr<Vector &, Op>);
    static_assert(HasCoefficientWiseOperation<Op>);
    static_assert(std::is_constructible_v<Node, const Child &, const Op &>);
    static_assert(!std::is_constructible_v<Node, const Child &>);

    auto mutable_op = [offset](int value) mutable { return value + ++offset; };
    static_assert(!HasUnaryExpr<Vector &, decltype(mutable_op)>);
    static_assert(!HasCoefficientWiseOperation<decltype(mutable_op)>);
    auto move_only = [p = std::make_unique<int>(7)](int value) { return value + *p; };
    static_assert(std::is_invocable_v<const decltype(move_only) &, const int &>);
    static_assert(!HasUnaryExpr<Vector &, decltype(move_only)>);
    static_assert(!HasCoefficientWiseOperation<decltype(move_only)>);

    auto result = v.unary_expr(op);
    auto owned = result.to_owned();
    offset = 99;
    v(0) = 10;
    CHECK(result(0) == 17);
    CHECK(result(1) == 9);
    CHECK(owned(0) == 8);
    CHECK(owned(2) == 10);
    auto copied = owned;
    CHECK(copied(1) == 9);
}

TEST_CASE("callback_const_lvalue_overload_and_input", "[unary][callback]") {
    Vector v{1, 2, 3};
    Overloaded op{8};
    auto result = v.unary_expr(op);
    static_assert(std::is_same_v<decltype(result)::value_type, long>);
    op.offset = 100;
    CHECK(result(0) == 9);
    CHECK(result(2) == 11);
    auto owned = result.to_owned();
    CHECK(owned(1) == 10);

    // The callback sees a const lvalue even when the child computes a value.
    auto computed = (-v).unary_expr(Overloaded{4});
    static_assert(std::is_same_v<decltype(computed)::value_type, long>);
    CHECK(computed(0) == 3);
    auto temporary = Vector{3, 4, 5}.unary_expr(Overloaded{2});
    CHECK(temporary(0) == 5);
    auto const_child = std::as_const(v).unary_expr(Overloaded{3});
    CHECK(const_child(0) == 4);
}

TEST_CASE("scalar_callback_order_const_invocation_and_state", "[unary][callback]") {
    Vector v{1, 2, 3};
    Scalar scalar{10};
    Ordered op{4};
    auto right = unary::ScalarOperation(v.expression(), scalar, op);
    auto left = unary::ScalarOperation(scalar, v.expression(), op);
    static_assert(std::is_same_v<decltype(right)::value_type, long>);
    static_assert(std::is_same_v<decltype(left)::value_type, double>);
    op.offset = 100;
    scalar.value = 100;
    CHECK(right(0) == -5);
    CHECK(left(0) == 13.5);
    auto right_owned = right.make_owned();
    auto left_owned = left.make_owned();
    v(0) = 20;
    CHECK(right_owned(0) == -5);
    CHECK(left_owned(0) == 13.5);
    CHECK(right(0) == 14);
    CHECK(left(0) == -5.5);

    auto right_only = unary::ScalarOperation(v.expression(), Scalar{3}, RightOnly{});
    auto left_only = unary::ScalarOperation(Scalar{3}, v.expression(), LeftOnly{});
    CHECK(right_only(1) == -1);
    CHECK(left_only(1) == 1);

    auto captured = [offset = 5](const auto &a, const auto &b) { return a - b + offset; };
    using Op = decltype(captured);
    using Right = unary::ScalarOperation<const Child &, Op, int, true>;
    using Left = unary::ScalarOperation<const Child &, Op, int, false>;
    static_assert(!std::is_default_constructible_v<Op>);
    static_assert(std::is_constructible_v<Right, const Child &, int, const Op &>);
    static_assert(std::is_constructible_v<Left, int, const Child &, const Op &>);
    static_assert(!std::is_constructible_v<Right, const Child &, int>);
    static_assert(!std::is_constructible_v<Left, int, const Child &>);
    Right captured_right(v.expression(), 3, captured);
    Left captured_left(3, v.expression(), captured);
    CHECK(captured_right.make_owned()(1) == 4);
    CHECK(captured_left.make_owned()(1) == 6);
}

TEST_CASE("callback_reference_result_is_materialized", "[unary][callback]") {
    Vector v{1, 2, 3};
    auto computed = -v;
    auto right = unary::ScalarOperation(computed.expression(), Scalar{7}, ReferenceResult{});
    auto left = unary::ScalarOperation(Scalar{7}, computed.expression(), ReferenceResult{});
    auto coefficientwise = computed.unary_expr(ReferenceResult{});
    static_assert(std::is_same_v<std::invoke_result_t<const ReferenceResult &,
                                                     const int &, const Scalar &>,
                                 const int &>);
    static_assert(std::is_same_v<std::invoke_result_t<const ReferenceResult &,
                                                     const Scalar &, const int &>,
                                 const int &>);
    static_assert(std::is_same_v<decltype(right)::value_type, int>);
    static_assert(std::is_same_v<decltype(left)::value_type, int>);
    static_assert(std::is_same_v<decltype(coefficientwise)::value_type, int>);
    static_assert(std::is_same_v<decltype(right(0)), int>);
    static_assert(std::is_same_v<decltype(left(0)), int>);
    static_assert(std::is_same_v<decltype(coefficientwise(0)), int>);

    // Extend the returned values' lifetimes, not references to expired child temporaries.
    const auto &right_value = right(0);
    const auto &left_value = left(1);
    const auto &coefficient_value = coefficientwise(2);
    auto right_owned = right.make_owned();
    auto left_owned = left.make_owned();
    v(0) = 9;
    v(1) = 8;
    v(2) = 7;
    CHECK(right_value == -1);
    CHECK(left_value == -2);
    CHECK(coefficient_value == -3);
    CHECK(right(0) == -9);
    CHECK(left(1) == -8);
    CHECK(coefficientwise(2) == -7);
    CHECK(right_owned(0) == -1);
    CHECK(left_owned(1) == -2);
}
